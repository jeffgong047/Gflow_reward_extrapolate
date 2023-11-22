import math
import numpy as np
import pandas as pd

from scipy.special import gammaln
from collections import namedtuple
import sys
import networkx as nx
from collections import namedtuple
from abc import ABC, abstractmethod
from copy import deepcopy
from pgmpy.metrics import structure_score
from pgmpy.base import DAG
LocalScore = namedtuple('LocalScore', ['key', 'score', 'prior'])



class BasePrior(ABC):
    """Base class for the prior over graphs p(G).

    Any subclass of `BasePrior` must return the contribution of log p(G) for a
    given variable with `num_parents` parents. We assume that the prior is modular.

    Parameters
    ----------
    num_variables : int (optional)
        The number of variables in the graph. If not specified, this gets
        populated inside the scorer class.
    """
    def __init__(self, num_variables=None):
        self._num_variables = num_variables
        self._log_prior = None

    def __call__(self, num_parents):
        return self.log_prior[num_parents]

    @property
    @abstractmethod
    def log_prior(self):
        pass

    @property
    def num_variables(self):
        if self._num_variables is None:
            raise RuntimeError('The number of variables is not defined.')
        return self._num_variables

    @num_variables.setter
    def num_variables(self, value):
        self._num_variables = value



class UniformPrior(BasePrior):
    @property
    def log_prior(self):
        if self._log_prior is None:
            self._log_prior = np.zeros((self.num_variables,))
        return self._log_prior


class ErdosRenyiPrior(BasePrior):
    def __init__(self, num_variables=None, num_edges_per_node=1.):
        super().__init__(num_variables)
        self.num_edges_per_node = num_edges_per_node

    @property
    def log_prior(self):
        if self._log_prior is None:
            num_edges = self.num_variables * self.num_edges_per_node  # Default value
            p = num_edges / ((self.num_variables * (self.num_variables - 1)) // 2)
            all_parents = np.arange(self.num_variables)
            self._log_prior = (all_parents * math.log(p)
                               + (self.num_variables - all_parents - 1) * math.log1p(-p))
        return self._log_prior


class EdgePrior(BasePrior):
    def __init__(self, num_variables=None, beta=1.):
        super().__init__(num_variables)
        self.beta = beta

    @property
    def log_prior(self):
        if self._log_prior is None:
            self._log_prior = np.arange(self.num_variables) * math.log(self.beta)
        return self._log_prior


class FairPrior(BasePrior):
    @property
    def log_prior(self):
        if self._log_prior is None:
            all_parents = np.arange(self.num_variables)
            self._log_prior = (
                    - gammaln(self.num_variables + 1)
                    + gammaln(self.num_variables - all_parents + 1)
                    + gammaln(all_parents + 1)
            )
        return self._log_prior





class BaseScore(ABC):
    """Base class for the scorer.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.

    prior : `BasePrior` instance
        The prior over graphs p(G).
    """
    def __init__(self, data, prior):
        self.data = data
        self.prior = prior
        self.column_names = list(data.columns)
        self.num_variables = len(self.column_names)
        self.prior.num_variables = self.num_variables

    def __call__(self, index, in_queue, out_queue, error_queue):
        try:
            while True:
                data = in_queue.get()
                if data is None:
                    break

                target, indices, indices_after = data
                local_score_before, local_score_after = self.get_local_scores(
                    target, indices, indices_after=indices_after)

                out_queue.put((True, *local_score_after))
                if local_score_before is not None:
                    out_queue.put((True, *local_score_before))

        except (KeyboardInterrupt, Exception):
            error_queue.put((index,) + sys.exc_info()[:2])
            out_queue.put((False, None, None, None))

    @abstractmethod
    def get_local_scores(self, target, indices, indices_after=None):
        pass


class BasePrior(ABC):
    """Base class for the prior over graphs p(G).

    Any subclass of `BasePrior` must return the contribution of log p(G) for a
    given variable with `num_parents` parents. We assume that the prior is modular.

    Parameters
    ----------
    num_variables : int (optional)
        The number of variables in the graph. If not specified, this gets
        populated inside the scorer class.
    """
    def __init__(self, num_variables=None):
        self._num_variables = num_variables
        self._log_prior = None

    def __call__(self, num_parents):
        return self.log_prior[num_parents]

    @property
    @abstractmethod
    def log_prior(self):
        pass

    @property
    def num_variables(self):
        if self._num_variables is None:
            raise RuntimeError('The number of variables is not defined.')
        return self._num_variables

    @num_variables.setter
    def num_variables(self, value):
        self._num_variables = value

StateCounts = namedtuple('StateCounts', ['key', 'counts'])


class BDeScore_gflow(BaseScore):
    """BDe score.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the (discrete) dataset D. Each column
        corresponds to one variable. If there is interventional data, the
        interventional targets must be specified in the "INT" column (the
        indices of interventional targets are assumed to be 1-based).

    prior : `BasePrior` instance
        The prior over graphs p(G).

    equivalent_sample_size : float (default: 1.)
        The equivalent sample size (of uniform pseudo samples) for the
        Dirichlet hyperparameters. The score is sensitive to this value,
        runs with different values might be useful.
    """
    def __init__(self, data, prior, equivalent_sample_size=1.):
        if 'INT' in data.columns:  # Interventional data
            # Indices should start at 0, instead of 1;
            # observational data will have INT == -1.
            self._interventions = data.INT.map(lambda x: int(x) - 1)
            data = data.drop(['INT'], axis=1)
        else:
            self._interventions = np.full(data.shape[0], -1)
        self.equivalent_sample_size = equivalent_sample_size
        super().__init__(data, prior)

        self.state_names = {
            column: sorted(self.data[column].cat.categories.tolist())
            for column in self.data.columns
        }
        self.node_to_index = {node:idx  for idx, node in enumerate(data.columns)}

    def get_local_scores(self, target, indices, indices_after=None):
        # Get all the state counts
        state_counts_before, state_counts_after = self.state_counts(
            target, indices, indices_after=indices_after)

        local_score_after = self.local_score(*state_counts_after)
        if state_counts_before is not None:
            local_score_before = self.local_score(*state_counts_before)
        else:
            local_score_before = None

        return (local_score_before, local_score_after)


    def get_sample_score(self, graph):
        '''

        :param graph: bayesian networks in networkx.Digraph format
        :return:
        '''
        log_graph_score = 0
        for node in graph.nodes():
            parents = [self.node_to_index[x] for x in graph.predecessors(node)]
            local_score = 0
            if parents:
                node_index = self.node_to_index[node]
                state_counts_before, state_counts_after = self.state_counts(
                    node_index, parents)
                local_score_before = self.local_score(*state_counts_before).score
                local_score_after = self.local_score(*state_counts_after).score
                log_graph_score += local_score_after - local_score_before
            print('node :',node, 'parents:',list(graph.predecessors(node)) ,'local score: ',local_score)
        return log_graph_score

    def state_counts(self, node_index, parents):
        # Source: pgmpy.estimators.BaseEstimator.state_counts()
        parents = [self.column_names[index] for index in parents]
        variable = self.column_names[node_index]

        data = self.data[self._interventions != node_index]
        data = data[[variable] + parents].dropna()
        state_count_data = (data.groupby([variable] + parents)
                            .size()
                            .unstack(parents))
        if not isinstance(state_count_data.columns, pd.MultiIndex):
            state_count_data.columns = pd.MultiIndex.from_arrays(
                [state_count_data.columns]
            )

        parent_states = [self.state_names[parent] for parent in parents]
        columns_index = pd.MultiIndex.from_product(parent_states, names=parents)

        state_counts_after = StateCounts(
            key=(node_index, tuple(parents)),
            counts=(state_count_data
                    .reindex(index=self.state_names[variable], columns=columns_index)
                    .fillna(0))
        )


        state_count_data = state_counts_after.counts.sum(axis=1).to_frame()

        state_counts_before = StateCounts(
            key=(node_index, ()),
            counts=state_count_data
        )


        return state_counts_before , state_counts_after

    def local_score(self, key, counts):
        counts = np.asarray(counts)
        num_parents_states = counts.shape[1]
        num_parents = len(key[1])

        log_gamma_counts = np.zeros_like(counts, dtype=np.float_)
        alpha = self.equivalent_sample_size / num_parents_states
        beta = self.equivalent_sample_size / counts.size

        # Compute log(gamma(counts + beta))
        gammaln(counts + beta, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=np.float_)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        local_score = (
                np.sum(log_gamma_counts)
                - np.sum(log_gamma_conds)
                + num_parents_states * math.lgamma(alpha)
                - counts.size * math.lgamma(beta)
        )

        return LocalScore(
            key=key,
            score=local_score,
            prior=self.prior(num_parents)
        )


def get_prior(name, **kwargs):
    prior = {
        'uniform': UniformPrior,
        'erdos_renyi': ErdosRenyiPrior,
        'edge': EdgePrior,
        'fair': FairPrior
    }
    return prior[name](**kwargs)



class BDeu(ABC):
    def __init__(self,evidences):
        self.evidences = evidences
    def estimate(self,  sample):
        evidences = self.evidences
        nodes = evidences['nodes']
        evidence_data = evidences['evidence_data']
        model = DAG()
        model.add_nodes_from(nodes = nodes)
        edges = [(element[0],element[1]) for element in sample]
        model.add_edges_from(edges)
        if model.edges():
            score = structure_score(model, evidence_data, scoring_method="bdeu")
        else:
            score= 0
        return score


class proxy_reward_function(ABC):
    def __init__(self,scorer_name, evidences):
        self.scorer_collection = {'BDeu':BDeu}
        self.evidences = evidences
        self.scorer = self.scorer_collection[scorer_name](evidences)

    def annotate(self,samples):
        rewards = []
        for s in samples:
            score = self.scorer.estimate(s)
            rewards.append(score)
        return rewards


