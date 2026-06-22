import torch
import torch.optim as optim
import random

def rcam_loss_reward(g, reward):
    device = g.device
    reward = torch.tensor(reward, device=device, dtype=torch.float)
    if reward >= 0:
        loss = -torch.norm(g, p=1)
    else:
        loss = torch.norm(g, p=2)
    return loss

class REINFROCE(object):
    def __init__(self, policy, target_policy, learning_rate=0.005, gamma=0.99, batch_size=32, num_batches=128, max_memory=10000):
        super(REINFROCE, self).__init__()
        self.policy = policy
        self.target_policy = target_policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma=gamma
        self.experience_buffer = []
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.max_memory = max_memory
    def memory_data(self, data):
        self.experience_buffer.append(data)
        if len(self.experience_buffer)>self.max_memory:
            self.experience_buffer.pop(0)
    def learn(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        experience = self.experience_buffer.copy()
        indices=list(range(len(experience)))
        random.shuffle(indices)

        batch_count=0
        for i in range(0,len(self.experience_buffer),self.batch_size):
            self.optimizer.zero_grad()
            total_loss=0
            if batch_count>=self.num_batches:
                break
            batch_indices=indices[i:i+self.batch_size]
            discounted_reward_t=0
            for idx in batch_indices:
                state,action,reward,next_state,done=self.experience_buffer[idx]
                current_prob, mask = self.policy(state.x.to(device),state.edge_index.to(device), state.weight.to(device))
                current_q_values = current_prob[0][action]
                next_action_prob, _=self.target_policy(next_state.x.to(device),next_state.edge_index.to(device),
                                                    next_state.weight.to(device))
                next_action=torch.argmax(next_action_prob,dim=-1)
                next_q_values=next_action_prob[0][next_action]
                discounted_reward_t=reward+self.gamma*next_q_values*(1-int(done))
                loss=(current_q_values-discounted_reward_t).pow(2)
                loss_causal = rcam_loss_reward(mask, reward)
                total_loss+=loss + loss_causal
            total_loss = total_loss/len(batch_indices)
            total_loss.backward()
            self.optimizer.step()
            batch_count+=1