import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch
import pandas as pd
import pickle
from itertools import permutations
from torch_geometric.data import Data

def to_pyg_data(MPMNode, MPMEdge, state_index, MPMEdgeW):
    node_feature = []
    node_feature.append(MPMNode.iloc[state_index])
    for i in MPMEdge[state_index]:
        node_feature.append(MPMNode.iloc[i])
    edge_index = list(permutations([i for i in range(len(node_feature))], 2))[:len(node_feature) - 1]
    node_feature = torch.tensor(node_feature, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    weight = MPMEdgeW[state_index]
    weight = torch.tensor(weight, dtype=torch.float)
    state = Data(x=node_feature, edge_index=edge_index, weight=weight)
    return state

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        if hasattr(data, 'batch'):
            x = global_mean_pool(x, data.batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return x

class AIRLDiscriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, gamma=0.99):
        super(AIRLDiscriminator, self).__init__()
        self.gamma = gamma
        self.g_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.h_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action, next_state):
        g_input = torch.cat([state, action], dim=-1)
        g_val = self.g_net(g_input)
        h_state = self.h_net(state)
        h_next_state = self.h_net(next_state)
        f_val = g_val + self.gamma * h_next_state - h_state
        return f_val

def train_airl(graph_encoder, discriminator, expert_loader, novice_loader, optimizer, device, num_epochs=10):
    criterion = nn.BCEWithLogitsLoss()
    graph_encoder.train()
    discriminator.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for expert_batch, novice_batch in zip(expert_loader, novice_loader):
            optimizer.zero_grad()
            expert_states = expert_batch['state']
            expert_next_states = expert_batch['next_state']
            expert_actions = expert_batch['action']
            novice_states = novice_batch['state']
            novice_next_states = novice_batch['next_state']
            novice_actions = novice_batch['action']

            encoded_expert_states = graph_encoder(expert_states)
            encoded_expert_next_states = graph_encoder(expert_next_states)
            encoded_novice_states = graph_encoder(novice_states)
            encoded_novice_next_states = graph_encoder(novice_next_states)

            expert_actions_onehot = nn.functional.one_hot(expert_actions, num_classes=2).float().to(device)
            novice_actions_onehot = nn.functional.one_hot(novice_actions, num_classes=2).float().to(device)

            encoded_states = torch.cat([encoded_expert_states, encoded_novice_states], dim=0)
            encoded_next_states = torch.cat([encoded_expert_next_states, encoded_novice_next_states], dim=0)
            actions_onehot = torch.cat([expert_actions_onehot, novice_actions_onehot], dim=0)

            expert_labels = torch.ones(encoded_expert_states.size(0), 1, device=device)
            novice_labels = torch.zeros(encoded_novice_states.size(0), 1, device=device)
            labels = torch.cat([expert_labels, novice_labels], dim=0)

            logits = discriminator(encoded_states, actions_onehot, encoded_next_states)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_losses.append(epoch_loss)
    return epoch_losses

def load_datasets_from_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    n = len(data)
    novice_dataset = data[:int(n * fraction)]
    expert_dataset = data[int(n * (1-fraction)):]
    return novice_dataset, expert_dataset
def convert_tuple_to_sample(t, MPMNode, MPMEdge, MPMEdgeW):
    state_index, action, next_state_index = t
    state_data = to_pyg_data(MPMNode, MPMEdge, state_index, MPMEdgeW)
    next_state_data = to_pyg_data(MPMNode, MPMEdge, next_state_index, MPMEdgeW)
    return {
        'state': state_data,
        'action': torch.tensor(action, dtype=torch.long),
        'next_state': next_state_data,
        'state_index': int(state_index),
        'next_state_index': int(next_state_index)
    }

def main():
    pkl_file = r'DATA.pkl'
    novice_tuples, expert_tuples = load_datasets_from_pkl(pkl_file)
    novice_tuples = [t[:2] + t[3:] for t in novice_tuples]
    expert_tuples = [t[:2] + t[3:] for t in expert_tuples]
    fileNode = "data.csv"
    fileEdge = "data.pkl"
    fileEdgeW = "data.pkl"
    MPMNode = pd.read_csv(fileNode, header=0)
    with open(fileEdge, 'rb') as fE:
        MPMEdge = pickle.load(fE)
    with open(fileEdgeW, 'rb') as fEW:
        MPMEdgeW = pickle.load(fEW)

    novice_dataset = [convert_tuple_to_sample(t, MPMNode, MPMEdge, MPMEdgeW) for t in novice_tuples]
    expert_dataset = [convert_tuple_to_sample(t, MPMNode, MPMEdge, MPMEdgeW) for t in expert_tuples]

    batch_size = int
    expert_loader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=False)
    novice_loader = DataLoader(novice_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = int
    lr = float

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graph_encoder = GraphEncoder(in_channels=39, hidden_channels=int, out_channels=int).to(device)
    discriminator = AIRLDiscriminator(state_dim=int, action_dim=2,
                                      hidden_dim=int, gamma=int).to(device)
    optimizer = optim.Adam(list(graph_encoder.parameters()) + list(discriminator.parameters()), lr=lr)

    train_airl(graph_encoder, discriminator, expert_loader, novice_loader, optimizer, device, num_epochs)
    torch.save(discriminator.g_net.state_dict(), "output.pth")

if __name__ == '__main__':
    main()
