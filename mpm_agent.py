import pandas as pd
import pickle
from mpm_policy import graph_nn
from mpm_drl import REINFROCE
from mpm_env import graph_env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fileCoorLabel = "data.csv"
fileNode = "data.csv"
fileEdge = "data.pkl"
fileEdgeW = "data.pkl"
fileGAE = "data.csv"
fileSpatial = "data.pkl"
fileDistacne = "data.pkl"
g_p = float
learning_rate=float
episodes = int
len_episode = int
warm_episode = int
update_interval = int
gamma = float 
batch_size = int
num_batches = int
max_memory = int
policy_action_space=int

env = graph_env(fileCoorLabel=fileCoorLabel,fileNode=fileNode,fileEdge=fileEdge, fileEdgeW=fileEdgeW,fileGAE=fileGAE,
                fileSpatial=fileSpatial, fileDistance=fileDistacne,g_p=g_p)
policy = graph_nn(action_space=policy_action_space, input_dim=pd.read_csv(fileNode).shape[1]).to(device)
target_policy = graph_nn(action_space=policy_action_space, input_dim=pd.read_csv(fileNode).shape[1]).to(device)
target_policy.load_state_dict(policy.state_dict())
learner = REINFROCE(policy=policy, target_policy=target_policy,learning_rate=learning_rate, gamma=gamma,
                    batch_size=batch_size, num_batches=num_batches,max_memory=max_memory)

eps = 1
interaction_records = []
for episode in range(episodes):
    state_graph = env.reset()
    done = False
    step = 0
    while not done:
        action_prob, mask = policy(state_graph.x.to(device), state_graph.edge_index.to(device), state_graph.weight.to(device))
        eps_prob = [eps, 1-eps]
        elements = [True, False]
        result = random.choices(elements, eps_prob)[0]
        if result:
            action = torch.randint(0,action_prob.size(-1),(1,))
        else:
            action = torch.argmax(action_prob, dim=-1)
        head_index = env.state_index
        next_state, reward, done, _ = env.step(action.item())
        end_index = env.state_index
        if step>=len_episode:
            done=True
        learner.memory_data((state_graph, action, reward, next_state, done))
        interaction_records.append((head_index, action.item(), reward, end_index))
        state_graph = next_state
        step += 1
    if episode > warm_episode:
        learner.learn()
    if episode%update_interval == 0 and episode!=0:
        target_policy.load_state_dict(policy.state_dict())
    if episode < 0.9*episodes:
        eps = 1 - (episode/episodes) * 0.95
    else:
        eps = 0.05

#data for IRL
with open('output.pkl', 'wb') as f:
    pickle.dump(interaction_records, f)