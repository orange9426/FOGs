import env as env_module
import solver as solver_module
import torch
import torch.nn as nn
import random
import tqdm

from solver.cfr.cfr import DepthLimited_CFR
from policy.policy import TabularPolicy
from policy.exploitability import exploitability

env = 'KuhnPoker'
buffer_capacity = 100
batch_size = 16
lr = 0.01
layers_sizes = [128, 128]


class ReplayBuffer(object):
    def __init__(self, replay_buffer_capacity):
        self._replay_buffer_capacity = replay_buffer_capacity
        self._data = []
        self._next_entry_index = 0

    def add(self, element):
        if len(self._data) < self._replay_buffer_capacity:
            self._data.append(element)
        else:
            self._data[self._next_entry_index] = element
            self._next_entry_index += 1
            self._next_entry_index %= self._replay_buffer_capacity

    def sample(self, num_samples):
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
                num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class MLP(nn.Module):
    def __init__(self, dinp, layer_size, dout):
        super(MLP, self).__init__()
        self.layers_sizes = [dinp] + layer_size + [dout]
        self.fcs = []
        for i in range(len(self.layers_sizes)-1):
            self.fcs.append(
                nn.Linear(self.layers_sizes[i], self.layers_sizes[i+1]))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x):
        for i in range(len(self.layers_sizes)-1):
            x = self.fcs[i](x)
            if i < len(self.layers_sizes)-2:
                x = nn.functional.relu(x)
        return x


class ReBeL(object):
    def __init__(self,
                 env,
                 buffer_capacity=buffer_capacity,
                 batch_size=batch_size,
                 lr=lr,
                 min_buffer_size=32,
                 layers_sizes=layers_sizes,
                 max_depth=2,
                 iteration_num=100,
                 learning_every=32):
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.game = getattr(env_module, env)()
        self.current_pbs = self.game.initial_pbs()
        self.max_depth = max_depth
        self.count = 0
        self.learning_every = learning_every
        self.min_buffer_size = min_buffer_size
        self.lr = lr
        self.iteration_num = iteration_num
        self.policy = TabularPolicy(self.game)

        dinp = self.game.get_tensor(self.current_pbs).size()[0]
        dout = len(self.current_pbs.prob_dict)
        self.value_net = MLP(dinp, layers_sizes, dout)
        self.batch_size = batch_size

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=self.lr)

    def step(self):
        solver = DepthLimited_CFR(self.game,
                                  self.value_net,
                                  self.current_pbs,
                                  max_depth=self.max_depth,
                                  iteration_num=self.iteration_num)
        policy_sub = solver.train_policy()
        self.policy.set_subgame_policy(policy_sub)
        self.replay_buffer.add(solver.get_training_data())
        self.current_pbs = solver.sample_pbs()
        self.count += 1

        if self.count % self.learning_every == 0:
            self.learn()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size or \
                len(self.replay_buffer) < self.min_buffer_size:
            return None

        data = self.replay_buffer.sample(self.batch_size)
        pbses = torch.tensor([t[0].tolist() for t in data])
        values = torch.tensor([t[1].tolist() for t in data])
        res = self.value_net(pbses)

        l = self.loss(res, values)
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

    def reset_episode(self):
        if self.current_pbs.is_terminal():
            self.current_pbs = self.game.initial_pbs()


def main():
    agents = ReBeL(env)
    episode_num = 500
    for ep in tqdm.tqdm(range(episode_num)):
        while not agents.current_pbs.is_terminal():
            agents.step()
        agents.reset_episode()
        if (ep+1) % 50 == 0:
            expl = exploitability(agents.game, agents.policy)
            print(expl) 
    #print(agents.policy.action_probabilities_table)
    agents.policy.print()


if __name__ == '__main__':
    main()
