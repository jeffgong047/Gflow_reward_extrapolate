import sys
from collections import namedtuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from configs import EnvConfig
from simple_parsing import ArgumentParser

from gfn.containers.states import correct_cast
from gfn.estimators import LogEdgeFlowEstimator
from gfn.losses import FMParametrization
from gfn.modules import Tabular, Uniform
from gfn.utils import validate
from dynamic_programming import backward

class Gflow(ABC):
    @abstractmethod
    def edge_flow_to_prob(self, edge_flow):
        pass

class gflow_extrapolate(Gflow):
    '''
    Base class for gflow network variant that specifies in extrapolating unseen rewards

    Parameters: samples

    '''

    def __init__(self, env):
        self.env = env
        self.model = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n)
        )

    def edge_flow_to_prob(self, edge_flow):
        '''
        convert from
        :param edge_flow:
        :return:
        '''
        edge_prob = torch.softmax(edge_flow, dim=-1)
        return edge_prob

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        edge_flow = self.model(state)
        edge_prob = self.edge_flow_to_prob(edge_flow)
        return edge_prob

    def extrapolate(self, state):
        pass

    def fit(self, batch):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        for state, action, reward, next_state, done in batch:
            optimizer.zero_grad()
            prediction = self.predict(state)
            target = torch.tensor([action], dtype=torch.int64)
            loss = loss_fn(prediction.unsqueeze(0), target)
            loss.backward()
            optimizer.step()

    def weight_flows(self):
        pass

    def backward_decomposition_base(self,samples):
        backward(samples)
        return

    def backward_decomposition_lowEntropy(self):
        pass

    def backward_decomposition_highEntropy(self):
        pass

