import os
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch_geometric.loader import DataLoader

# --- Global Parameters ---
TRAIN_DATA_FILE = 'CyberAI2025_train.data'
TRAIN_LABEL_FILE = 'CyberAI2025_train.label'
EMBEDDING_FILE = 'node_embeddings_p1.2_q1.0.kv'  # Best p/q value from tuning
MODEL_SAVE_PATH = 'ensemble_model_C_simple_gcn.pth' 
LEARNING_RATE = 0.001 
EPOCHS = 300 

# --- 1. Data Processing for GNN (Fixed Embeddings) ---
def parse_and_create_pyg_data_with_embeddings(data_file, labels_map, keyed_vectors, y_labels=None):
    # (Function body is the same as the original 'gnnpart2_v2.py' logic)
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
            node_mapping = {node: i for i, node in enumerate(unique_nodes)}
            
            edge_list = []
            for caller, callee in G.edges():
                edge_list.append([node_mapping[caller], node_mapping[callee]])

            if not edge_list:
                continue

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            node_features_list = []
            for node_name in unique_nodes:
                try:
                    embedding = keyed_vectors[node_name]
                except KeyError:
                    embedding = np.zeros(keyed_vectors.vector_size)

                in_degree = G.in_degree(node_name)
                out_degree = G.out_degree(node_name)
                combined_features = np.concatenate([embedding, [in_degree, out_degree]])
                node_features_list.append(combined_features)
            
            node_features = torch.tensor(np.array(node_features_list), dtype=torch.float)

            if y_labels:
                label = torch.tensor([labels_map[y_labels[i]]], dtype=torch.long)
            else:
                label = None
                
            data = Data(x=node_features, edge_index=edge_index, y=label)
            graphs_list.append(data)
    
    return graphs_list

# --- 2. GCN Model Definition (Simplest Architecture) ---
class GCN_Simple(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN_Simple, self).__init__()
        # Simplified 2-layer GCN
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.linear = Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

# --- 3. Final Training and Saving ---
def train_and_save_final_model(all_train_graphs, num_node_features, num_classes):
    
    model = GCN_Simple(num_node_features=num_node_features, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader = DataLoader(all_train_graphs, batch_size=32, shuffle=True)

    print(f"Training final Model C (Simple GCN) on ALL {len(all_train_graphs)} samples for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, data.y.view(-1)) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {total_loss / len(train_loader):.4f}")

    # --- SAVE THE FINAL MODEL STATE ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel C weights saved successfully to: {MODEL_SAVE_PATH}")


# --- Main Execution ---
if __name__ == '__main__':
    print("--- Preparing Model C (Simple GCN) for Ensemble ---")

    # Load pre-trained embeddings
    keyed_vectors = KeyedVectors.load(EMBEDDING_FILE)
    embedding_dim = keyed_vectors.vector_size
    num_node_features = embedding_dim + 2

    # Load labels for map creation
    y_full_train = [line.strip() for line in open(TRAIN_LABEL_FILE, 'r')]
    labels_map = {label: i for i, label in enumerate(sorted(list(set(y_full_train))))}
    num_classes = len(labels_map)

    # Parse data for training
    all_train_graphs = parse_and_create_pyg_data_with_embeddings(TRAIN_DATA_FILE, labels_map, keyed_vectors, y_full_train)
    all_train_graphs = [g for g in all_train_graphs if g is not None]
    
    train_and_save_final_model(all_train_graphs, num_node_features, num_classes)