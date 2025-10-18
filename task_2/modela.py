import os
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear, Embedding
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# --- Global Parameters ---
TRAIN_DATA_FILE = 'CyberAI2025_train.data'
TRAIN_LABEL_FILE = 'CyberAI2025_train.label'
# FINAL MODEL WEIGHTS WILL BE SAVED HERE
MODEL_SAVE_PATH = 'ensemble_model_A_trainable_gat.pth' 
EMBEDDING_DIM = 64
LEARNING_RATE = 0.001
EPOCHS = 300 # Keeping the high epoch count for best convergence

# Global mapping for all unique functions (needed for nn.Embedding)
GLOBAL_FUNC_TO_ID = {}
GLOBAL_FUNC_ID_COUNTER = 0

def get_node_id(name):
    """Assigns a unique integer ID to every unique function name."""
    global GLOBAL_FUNC_ID_COUNTER
    if name not in GLOBAL_FUNC_TO_ID:
        GLOBAL_FUNC_TO_ID[name] = GLOBAL_FUNC_ID_COUNTER
        GLOBAL_FUNC_ID_COUNTER += 1
    return GLOBAL_FUNC_TO_ID[name]

# --- 1. Data Processing for GNN ---
def parse_and_create_pyg_data(data_file, labels_map, y_labels=None):
    # (Function body remains the same as your previous script)
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
            
            node_ids = torch.tensor([get_node_id(name) for name in unique_nodes], dtype=torch.long)
            
            degrees = []
            for name in unique_nodes:
                degrees.append([G.in_degree(name), G.out_degree(name)])
            degree_features = torch.tensor(degrees, dtype=torch.float)

            node_mapping = {name: i for i, name in enumerate(unique_nodes)}
            edge_list = []
            for caller, callee in G.edges():
                edge_list.append([node_mapping[caller], node_mapping[callee]])

            if not edge_list:
                continue

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            label = torch.tensor([labels_map[y_labels[i]]], dtype=torch.long)
            
            data = Data(x=node_ids, edge_index=edge_index, y=label, degrees=degree_features)
            graphs_list.append(data)
    
    return graphs_list


# --- 2. GAT Model with Trainable Embedding Layer ---
class GAT_Trainable(torch.nn.Module):
    def __init__(self, num_unique_functions, embedding_dim, num_classes, heads=4, dropout=0.6):
        super(GAT_Trainable, self).__init__()
        
        # 1. TRAINABLE EMBEDDING LAYER
        self.embedding = Embedding(num_unique_functions, embedding_dim)
        initial_feature_dim = embedding_dim + 2
        
        self.conv1 = GATConv(initial_feature_dim, 64, heads=heads, dropout=dropout)
        self.conv2 = GATConv(64 * heads, 64, heads=heads, dropout=dropout)
        self.conv3 = GATConv(64 * heads, 64, heads=1, concat=False, dropout=dropout)
        self.linear = Linear(64, num_classes)

    def forward(self, data):
        x_ids, edge_index, batch = data.x, data.edge_index, data.batch
        
        x_emb = self.embedding(x_ids)
        x_combined = torch.cat([x_emb, data.degrees], dim=1)
        
        x = self.conv1(x_combined, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x


# --- 3. Final Training and Saving ---
def train_and_save_final_model(all_train_graphs):
    
    # Initialize GAT with the specific trainable architecture
    model = GAT_Trainable(
        num_unique_functions=GLOBAL_FUNC_ID_COUNTER, 
        embedding_dim=EMBEDDING_DIM, 
        num_classes=len(labels_map), 
        heads=4, 
        dropout=0.6
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Use the full dataset for the final training loader
    train_loader = DataLoader(all_train_graphs, batch_size=32, shuffle=True)

    print(f"Training final Model A (Trainable GAT) on ALL {len(all_train_graphs)} samples for {EPOCHS} epochs...")
    
    # Training loop
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
        
        # Print a checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {total_loss / len(train_loader):.4f}")

    # --- SAVE THE FINAL MODEL STATE ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel A weights saved successfully to: {MODEL_SAVE_PATH}")
    
    # Return the trained model for immediate use in the ensemble script (if needed)
    return model


# --- Main Execution ---
if __name__ == '__main__':
    print("--- Preparing Model A (Trainable GAT) for Ensemble ---")

    # Load labels first for map creation
    y_full_train = [line.strip() for line in open(TRAIN_LABEL_FILE, 'r')]
    labels_map = {label: i for i, label in enumerate(sorted(list(set(y_full_train))))}

    # Parse data. This builds the GLOBAL_FUNC_TO_ID dictionary as a side effect.
    all_train_graphs = parse_and_create_pyg_data(TRAIN_DATA_FILE, labels_map, y_full_train)
    
    print(f"Total unique functions found: {GLOBAL_FUNC_ID_COUNTER}")
    
    train_and_save_final_model(all_train_graphs)