import gym
import torch
from torch_geometric.data import Data
from itertools import permutations
import pandas as pd
import pickle
import numpy as np

class graph_env(gym.Env):
    def __init__(self, fileCoorLabel,fileNode, fileEdge, fileEdgeW, fileGAE, fileSpatial, fileDistance,g_p):
        super(graph_env, self).__init__()
        self.MPMNode = pd.read_csv(fileNode, header=0)
        self.fileCoorLabel = pd.read_csv(fileCoorLabel)
        self.label_1 = self.fileCoorLabel[self.fileCoorLabel['label']!= labelnum]
        self.label_0 = self.fileCoorLabel[self.fileCoorLabel['label']== labelnum]
        with open(fileEdge, 'rb') as fE:
            self.MPMEdge = pickle.load(fE)
        with open(fileEdgeW, 'rb') as fEW:
            self.MPMEdgeW = pickle.load(fEW)
        with open(fileSpatial, 'rb') as fS:
            self.data_index_pkl = pickle.load(fS)
        with open(fileDistance, 'rb') as fD:
            self.property_distace = pickle.load(fD)
        self.reward_GAE = pd.read_csv(fileGAE, header=0)
        self.g_p = g_p
        self.data_index = 0
    def to_pyg_data(self, state_index):
        node_feature = []
        node_feature.append(self.MPMNode.iloc[state_index])
        for i in self.MPMEdge[state_index]:
            node_feature.append(self.MPMNode.iloc[i])
        edge_index=list(permutations([i for i in range(len(node_feature))],2))[:len(node_feature)-1]

        node_feature = torch.tensor(node_feature, dtype=torch.float)
        edge_index = torch.tensor(edge_index,dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        weight = self.MPMEdgeW[state_index]
        weight = torch.tensor(weight, dtype=torch.float)
        state = Data(x=node_feature, edge_index=edge_index, weight=weight)
        return state
    def reset(self):
        random_index = self.label_1.sample(n=1).index[0]
        self.state_index = random_index
        state = self.to_pyg_data(self.state_index)
        return state
    def step(self, action):
        reward = self.get_reward(action)
        state = self.next_state(action)
        done = False
        others = {}
        return state, reward, done, others

    def get_reward(self, action):
        if self.reward_GAE['label1'].iloc[self.state_index] != labelnum:
            r_GAE=self.reward_GAE['normalisation'].iloc[self.state_index]
            if action == 1:
                r_deposit = float
            else:
                r_deposit = float
        else:
            r_GAE = self.reward_GAE['normalisation'].iloc[self.state_index]
            if action == 0:
                r_deposit = float
            else:
                r_deposit = float

        return r_deposit+r_GAE
    def next_state(self, action):
        index=np.random.choice(['d_l','d_u'], p=[self.g_p,1-self.g_p])
        if index=='d_l':
            random_index=self.label_1.sample(n=1).index[0]
            self.state_index=random_index
            state=self.to_pyg_data(self.state_index)
        else:
            sorted_neighbors = np.argsort(self.property_distace[self.state_index])
            if action == 1:
                _index = sorted_neighbors[-int:]
                max_index = np.random.choice(_index)
                self.state_index = self.data_index_pkl[self.state_index][max_index]
                state = self.to_pyg_data(self.state_index)
            else:
                _index = sorted_neighbors[int:]
                min_index = np.random.choice(_index)
                self.state_index = self.data_index_pkl[self.state_index][min_index]
                state = self.to_pyg_data(self.state_index)
        return state