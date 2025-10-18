import os
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch.nn import Linear, Embedding
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import KeyedVectors
from torch_geometric.loader import DataLoader

# --- Global Configuration ---
TRAIN_DATA_FILE = 'CyberAI2025_train.data'
TRAIN_LABEL_FILE = 'CyberAI2025_train.label'
EMBEDDING_FILE = 'node_embeddings_p1.2_q1.0.kv' # Best Node2Vec features

# --- Model Weights Paths (The three models you saved) ---
MODEL_A_PATH = 'ensemble_model_A_trainable_gat.pth' # Trainable GAT
MODEL_B_PATH = 'ensemble_model_B_fixed_gat.pth'     # Fixed GAT
MODEL_C_PATH = 'ensemble_model_C_simple_gcn.pth'    # Simple GCN

# --- Global Parameters (Used for model initialization) ---
K_FOLDS = 5
EMBEDDING_DIM = 64
MODEL_A_CHECKPOINT_SIZE = 6072 # Final size of Model A's embedding table
SAFE_UNKNOWN_ID = 0 
NUM_FIXED_FEATURES = EMBEDDING_DIM + 2 

# --- Global Mappings & Utilities ---
GLOBAL_FUNC_TO_ID_TRAIN = {} 
GLOBAL_FUNC_ID_COUNTER_TRAIN = 0

def get_node_id_train_only(name):
    global GLOBAL_FUNC_ID_COUNTER_TRAIN
    if name not in GLOBAL_FUNC_TO_ID_TRAIN:
        GLOBAL_FUNC_TO_ID_TRAIN[name] = GLOBAL_FUNC_ID_COUNTER_TRAIN
        GLOBAL_FUNC_ID_COUNTER_TRAIN += 1
    return GLOBAL_FUNC_TO_ID_TRAIN[name]

# --- 1. Model Definitions (MUST be identical to saved models) ---
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

# --- 2. Data Parsing Utility (Core for K-Fold) ---
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

def parse_all_train_data(data_file, labels_map, keyed_vectors, y_labels):
    """Parses ALL training data into a list of PyG Data objects with all features."""
    
    graphs_list = []
    
    with open(data_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            edges = line.strip().split(' ')
            G = nx.DiGraph()
            for edge in edges:
                if '#' in edge:
                    caller, callee = edge.split('#')
                    G.add_edge(caller, callee)

            unique_nodes = sorted(list(G.nodes()))
            
            # --- FIXED EMBEDDINGS (For Models B & C) ---
            fixed_features_list = []
            for node_name in unique_nodes:
                try:
                    embedding = keyed_vectors[node_name]
                except KeyError:
                    embedding = np.zeros(keyed_vectors.vector_size)

                in_degree = G.in_degree(node_name)
                out_degree = G.out_degree(node_name)
                fixed_combined = np.concatenate([embedding, [in_degree, out_degree]])
                fixed_features_list.append(fixed_combined)
            fixed_node_features = torch.tensor(np.array(fixed_features_list), dtype=torch.float)

            # --- TRAINABLE EMBEDDINGS (For Model A) ---
            trainable_ids = []
            for name in unique_nodes:
                 node_id = GLOBAL_FUNC_TO_ID_TRAIN.get(name, SAFE_UNKNOWN_ID)
                 trainable_ids.append(node_id)
            trainable_ids_tensor = torch.tensor(trainable_ids, dtype=torch.long)
            degrees_tensor = torch.tensor([[G.in_degree(name), G.out_degree(name)] for name in unique_nodes], dtype=torch.float)
            
            # Create a single Data object that holds ALL necessary features for all models
            label = torch.tensor([labels_map[y_labels[i]]], dtype=torch.long)
            node_mapping = {name: i for i, name in enumerate(unique_nodes)}
            edge_list = [[node_mapping[c], node_mapping[t]] for c, t in G.edges()]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            data = Data(
                x_fixed=fixed_node_features, 
                x_trainable=trainable_ids_tensor, 
                degrees=degrees_tensor,
                edge_index=edge_index, 
                y=label
            )
            graphs_list.append(data)
    
    return graphs_list

# --- 3. Ensemble Prediction Logic (The K-Fold Loop) ---
def ensemble_predict_kfold(models_info, all_graphs, y_full_train, num_fixed_features, num_classes):
    """Performs k-fold ensemble prediction on validation data."""
    
    # K-fold setup
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_f1_scores = []
    
    y_target_array = np.array([g.y.item() for g in all_graphs])
    full_indices = np.arange(len(all_graphs)) 

    print(f"Starting {K_FOLDS}-fold cross-validation for Ensemble...")
    
    for fold, (train_indices, val_indices) in enumerate(skf.split(full_indices, y_target_array)):
        print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")
        
        # --- 1. COLLECT TRUE LABELS FOR THE FOLD ---
        val_data = [all_graphs[i] for i in val_indices]
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        # Collect ALL true labels for the validation set (size 280)
        val_labels = []
        for data in val_loader:
             val_labels.extend(data.y.view(-1).tolist())
        
        # --- 2. ENSEMBLE PREDICTION LOOP ---
        all_probabilities = []

        for model_info in models_info:
            
            # --- Model Setup (Re-initialization for each model) ---
            model_class = GAT_Base if 'gat' in model_info['path'] else GCN_Simple
            
            if 'model_A' in model_info['path']:
                model = GAT_Base(num_fixed_features, num_classes, is_trainable=True, num_unique_functions=MODEL_A_CHECKPOINT_SIZE, embedding_dim=EMBEDDING_DIM)
            elif 'model_B' in model_info['path']:
                model = GAT_Base(num_fixed_features, num_classes)
            elif 'model_C' in model_info['path']:
                model = GCN_Simple(num_fixed_features, num_classes)
                
            model.load_state_dict(torch.load(model_info['path']))
            model.eval()
            
            model_probs = []
            with torch.no_grad():
                for data in val_loader:
                    # Prepare input based on model type (Model A needs x_trainable, B/C needs x_fixed)
                    if 'model_A' in model_info['path']:
                        data.x = data.x_trainable
                    else:
                        data.x = data.x_fixed
                        
                    logits = model(data)
                    probs = F.softmax(logits, dim=1)
                    model_probs.append(probs)
            
            all_probabilities.append(torch.cat(model_probs, dim=0))

        # --- 3. Soft Voting and Final Metrics ---
        ensemble_probs = sum(all_probabilities) / len(all_probabilities)
        final_preds = ensemble_probs.argmax(dim=1).tolist()
        
        # Calculate metrics using correctly sized lists (both size 280)
        accuracy = accuracy_score(val_labels, final_preds)
        f1_macro = f1_score(val_labels, final_preds, average='macro')
        
        fold_accuracies.append(accuracy)
        fold_f1_scores.append(f1_macro)
        
        print(f"Fold Accuracy: {accuracy:.4f}")
        print(f"Fold Macro F1: {f1_macro:.4f}")

    print("\n--- Final Ensemble Cross-Validation Results ---")
    print(f"Average Accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
    print(f"Average Macro F1: {np.mean(fold_f1_scores):.4f} +/- {np.std(fold_f1_scores):.4f}")

    return np.mean(fold_accuracies), np.mean(fold_f1_scores)


# --- Main Execution ---
if __name__ == '__main__':
    print("Loading pre-trained embeddings...")
    keyed_vectors = KeyedVectors.load(EMBEDDING_FILE)
    
    num_fixed_features = EMBEDDING_DIM + 2 # 64 + 2
    
    print("Preparing data for Ensemble K-Fold Validation...")
    
    _build_global_map_from_file(TRAIN_DATA_FILE)
    
    y_full_train = [line.strip() for line in open(TRAIN_LABEL_FILE, 'r')]
    labels_map = {label: i for i, label in enumerate(sorted(list(set(y_full_train))))}
    num_classes = len(labels_map)

    # 1. Parse all training data into PyG objects
    all_train_graphs = parse_all_train_data(TRAIN_DATA_FILE, labels_map, keyed_vectors, y_full_train)
    all_train_graphs = [g for g in all_train_graphs if g is not None]
    
    # --- CONFIGURE ENSEMBLE ---
    ensemble_models = [
        {'model': None, 'path': MODEL_A_PATH, 'loader': None},
        {'model': None, 'path': MODEL_B_PATH, 'loader': None},
        {'model': None, 'path': MODEL_C_PATH, 'loader': None},
    ]

    # --- EXECUTE ENSEMBLE K-FOLD VALIDATION ---
    ensemble_predict_kfold(ensemble_models, all_train_graphs, y_full_train, num_fixed_features, num_classes)