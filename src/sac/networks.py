import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from src.utils.probability import GumbelSoftmax


class ParentSAC(nn.Module):
    def __init__(self, name, chkpt_dir, input_dims, fc1_dims, fc2_dims):
        nn.Module.__init__(self)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(ParentSAC):
    def __init__(
            self,
            beta, input_dims, n_actions,
            fc1_dims=512,
            fc2_dims=512,
            optimizer=optim.Adam,
            name='critic',
            chkpt_dir='tmp/'
    ):
        super().__init__(name=name, chkpt_dir=chkpt_dir, input_dims=input_dims,
                         fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optimizer(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)
        return q


class ValueNetwork(ParentSAC):
    def __init__(
            self,
            beta, input_dims,
            fc1_dims=512,
            fc2_dims=512,
            optimizer=optim.Adam,
            name='value',
            chkpt_dir='tmp'
    ):
        super().__init__(name=name, chkpt_dir=chkpt_dir, input_dims=input_dims,
                         fc1_dims=fc1_dims, fc2_dims=fc2_dims)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optimizer(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        v = self.v(state_value)

        return v


class ActorNetwork(ParentSAC):
    def __init__(
            self,
            alpha, input_dims,
            fc1_dims=512,
            fc2_dims=512,
            n_reactants=1024,
            n_reactions=58,
            optimizer=optim.Adam,
            name='actor',
            chkpt_dir='tmp'
    ):
        super().__init__(name=name, chkpt_dir=chkpt_dir, input_dims=input_dims,
                         fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.n_reactants = n_reactants
        self.n_reactions = n_reactions
        self.name = name
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.reactant = nn.Linear(self.fc2_dims, self.n_reactants)
        self.reaction = nn.Linear(self.fc2_dims, self.n_reactions)
        self.optimizer = optimizer(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        prob_reactant = self.reactant(prob)
        prob_reaction = self.reaction(prob)
        return prob_reactant#, prob_reaction

    def sample_normal(self, state):
        output = self.forward(state)
        probabilities = GumbelSoftmax(logits=output, temperature=1)
        actions = probabilities.rsample()
        action = actions
        log_probs = probabilities.log_prob(actions)
        return action, log_probs
