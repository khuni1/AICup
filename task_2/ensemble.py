import os
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch.nn import Linear, Embedding
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch_geometric.loader import DataLoader

# --- Global Configuration ---
TEST_DATA_FILE = 'CyberAI2025_test.data'
TRAIN_LABEL_FILE = 'CyberAI2025_train.label'
TRAIN_DATA_FILE = 'CyberAI2025_train.data' 
EMBEDDING_FILE_C = 'node_embeddings_p1.2_q1.0.kv'
PREDICTIONS_FILE = 'final_ensemble_predictions.csv'

# --- Model Weights Paths ---
MODEL_A_PATH = 'ensemble_model_A_trainable_gat.pth'
MODEL_B_PATH = 'ensemble_model_B_fixed_gat.pth'     
MODEL_C_PATH = 'ensemble_model_C_simple_gcn.pth'    

# --- Global Mappings (Needed for Model A) ---
GLOBAL_FUNC_TO_ID_TRAIN = {} 
GLOBAL_FUNC_ID_COUNTER_TRAIN = 0
EMBEDDING_DIM = 64 

# The exact size Model A was trained with (Indices 0 to 6071)
MODEL_A_CHECKPOINT_SIZE = 6072
# Index 0 is the safest index; Index 6072 is the out-of-bounds index that causes the crash.
SAFE_UNKNOWN_ID = 0 

def get_node_id_train_only(name):
    """Assigns a unique ID ONLY for functions seen in training."""
    global GLOBAL_FUNC_ID_COUNTER_TRAIN
    if name not in GLOBAL_FUNC_TO_ID_TRAIN:
        GLOBAL_FUNC_TO_ID_TRAIN[name] = GLOBAL_FUNC_ID_COUNTER_TRAIN
        GLOBAL_FUNC_ID_COUNTER_TRAIN += 1
    return GLOBAL_FUNC_TO_ID_TRAIN[name]

# --- Model Definitions (Unchanged) ---
class GAT_Base(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, heads=4, dropout=0.6, is_trainable=False, num_unique_functions=0, embedding_dim=64):
        super().__init__()
        self.is_trainable = is_trainable
        initial_feature_dim = num_node_features
        if self.is_trainable:
            self.embedding = Embedding(num_unique_functions, embedding_dim)
            initial_feature_dim = embedding_dim + 2
        self.conv1 = GATConv(initial_feature_dim, 64, heads=heads, dropout=dropout)
        self.conv2 = GATConv(64 * heads, 64, heads=heads, dropout=dropout)
        self.conv3 = GATConv(64 * heads, 64, heads=1, concat=False, dropout=dropout)
        self.linear = Linear(64, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.is_trainable:
            x_emb = self.embedding(data.x)
            x = torch.cat([x_emb, data.degrees], dim=1)
        else:
            x = x
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.linear(x)

class GCN_Simple(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.linear = Linear(64, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.linear(x)


# --- UTILITY FUNCTION FOR MAP BUILDING ---
def _build_global_map_from_file(data_file):
    """Processes a file to populate the GLOBAL_FUNC_TO_ID_TRAIN map."""
    with open(data_file, 'r') as f:
        for line in f:
            edges = line.strip().split(' ')
            for edge in edges:
                if '#' in edge:
                    caller, callee = edge.split('#')
                    get_node_id_train_only(caller)
                    get_node_id_train_only(callee)
    return

# --- 2. Data Loading Utility (Adjusted for Test Data) ---
def parse_data_for_prediction(data_file, keyed_vectors_c):
    """Parses data using the final, fully-built global map."""
    fixed_graphs, trainable_graphs = [], []
    
    with open(data_file, 'r') as f:
        for line in f:
            edges = line.strip().split(' ')
            G = nx.DiGraph()
            for edge in edges:
                if '#' in edge:
                    caller, callee = edge.split('#')
                    G.add_edge(caller, callee)

            unique_nodes = sorted(list(G.nodes()))
            node_mapping = {name: i for i, name in enumerate(unique_nodes)}
            
            edge_list = []
            for caller, callee in G.edges():
                edge_list.append([node_mapping[caller], node_mapping[callee]])

            if not edge_list: continue

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            # --- FIXED EMBEDDINGS (For Models B & C) ---
            fixed_features_list = []
            for node_name in unique_nodes:
                try:
                    embedding = keyed_vectors_c[node_name]
                except KeyError:
                    embedding = np.zeros(keyed_vectors_c.vector_size)

                in_degree = G.in_degree(node_name)
                out_degree = G.out_degree(node_name)
                fixed_combined = np.concatenate([embedding, [in_degree, out_degree]])
                fixed_features_list.append(fixed_combined)
            
            fixed_node_features = torch.tensor(np.array(fixed_features_list), dtype=torch.float)

            # --- TRAINABLE EMBEDDINGS (For Model A) ---
            trainable_ids = []
            for name in unique_nodes:
                 # We force any unseen index back to the safe default (Index 0)
                 node_id = GLOBAL_FUNC_TO_ID_TRAIN.get(name, SAFE_UNKNOWN_ID)
                 trainable_ids.append(node_id)
                 
            degrees = torch.tensor([[G.in_degree(name), G.out_degree(name)] for name in unique_nodes], dtype=torch.float)
            
            data_fixed = Data(x=fixed_node_features, edge_index=edge_index)
            data_trainable = Data(x=torch.tensor(trainable_ids, dtype=torch.long), edge_index=edge_index, degrees=degrees)

            fixed_graphs.append(data_fixed)
            trainable_graphs.append(data_trainable)
            
    return fixed_graphs, trainable_graphs

# --- 3. Final Ensemble Prediction Logic ---
def ensemble_predict(models, test_loaders):
    """Loads weights and generates averaged Softmax probabilities."""
    all_probabilities = []
    
    with torch.no_grad():
        for model_info, loader in zip(models, test_loaders):
            model = model_info['model']
            
            # Load the saved weights
            model.load_state_dict(torch.load(model_info['path']))
            model.eval()
            
            model_probs = []
            for data in loader:
                logits = model(data)
                probs = F.softmax(logits, dim=1)
                model_probs.append(probs)
            
            all_probabilities.append(torch.cat(model_probs, dim=0))

    ensemble_probs = sum(all_probabilities) / len(all_probabilities)
    final_preds = ensemble_probs.argmax(dim=1).tolist()
    return final_preds

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Final Ensemble Prediction ---")
    
    # 1. --- BUILD GLOBAL NODE MAP (PASS 1) ---
    _build_global_map_from_file(TRAIN_DATA_FILE)
    
    # 2. --- PARSE TEST DATA ---
    # Load labels for final output mapping
    y_full_train = [line.strip() for line in open(TRAIN_LABEL_FILE, 'r')]
    labels_map = {label: i for i, label in enumerate(sorted(list(set(y_full_train))))}
    num_classes = len(labels_map)
    
    keyed_vectors_c = KeyedVectors.load(EMBEDDING_FILE_C)
    fixed_test_graphs, trainable_test_graphs = parse_data_for_prediction(TEST_DATA_FILE, keyed_vectors_c)
    
    # Calculate final feature dimensions
    embedding_dim = 64
    num_fixed_features = embedding_dim + 2 

    # --- DEFINE MODEL INSTANCES (Using CORRECT Sizes) ---
    # Model A: Trainable GAT (Initialized with the exact checkpoint size: 6072)
    model_A = GAT_Base(num_fixed_features, num_classes, is_trainable=True, num_unique_functions=MODEL_A_CHECKPOINT_SIZE, embedding_dim=embedding_dim)

    # Model B: Fixed GAT (architecture is simple)
    model_B = GAT_Base(num_fixed_features, num_classes, is_trainable=False)
    
    # Model C: Simple GCN
    model_C = GCN_Simple(num_fixed_features, num_classes)

    # --- DEFINE LOADERS ---
    trainable_loader = DataLoader(trainable_test_graphs, batch_size=32, shuffle=False)
    fixed_loader = DataLoader(fixed_test_graphs, batch_size=32, shuffle=False)

    # --- CONFIGURE ENSEMBLE ---
    ensemble_models = [
        {'model': model_A, 'path': MODEL_A_PATH, 'loader': trainable_loader},
        {'model': model_B, 'path': MODEL_B_PATH, 'loader': fixed_loader},
        {'model': model_C, 'path': MODEL_C_PATH, 'loader': fixed_loader},
    ]

    # --- EXECUTE ENSEMBLE PREDICTION ---
    print(f"Ensembling predictions from {len(ensemble_models)} models...")
    final_preds = ensemble_predict(ensemble_models, [m['loader'] for m in ensemble_models])

    # --- SAVE FINAL SUBMISSION ---
    id_to_label = {v: k for k, v in labels_map.items()}
    final_labels = [id_to_label[pred] for pred in final_preds]
    
    with open(PREDICTIONS_FILE, 'w') as f:
        for label in final_labels:
            f.write(f"{label}\n")

    print(f"\nSUCCESS! Final ensemble predictions saved to: {PREDICTIONS_FILE}")
    print("--- highest-confidence predictions for the competition ---")