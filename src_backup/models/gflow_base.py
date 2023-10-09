from abc import ABC
import jax.numpy as jnp

class GFlow(ABC):
    '''
    Base class for GFlow network variants that specialize in extrapolating unseen rewards.

    Parameters: samples

    Implementation of the GFlow is based on the interpretation that GFlow aims to obtain a generative model that can
    break down complicated probability distributions from which samples are generated. Such probability distributions
    are hard to infer directly from data, but GFlow net offers a perspective to leverage the sample generation structure,
    such that we have a structural generative model.
    '''

    def __init__(self, samples):
        self.samples = samples

    def train(self):
        '''
        Train the GFlow network on the given samples.

        This method should be implemented by the subclasses of GFlow.
        '''
        raise NotImplementedError("Subclasses should implement this method.")

    def generate(self, num_samples):
        '''
        Generate new samples using the GFlow network.

        Parameters:
        num_samples (int): Number of samples to generate.

        Returns:
        jnp.ndarray: A NumPy array of generated samples.

        This method should be implemented by the subclasses of GFlow.
        '''
        raise NotImplementedError("Subclasses should implement this method.")

    def evaluate(self, test_samples):
        '''
        Evaluate the GFlow network on a given set of test samples.

        Parameters:
        test_samples (jnp.ndarray): A NumPy array of test samples.

        Returns:
        float: An evaluation score for the GFlow network.

        This method should be implemented by the subclasses of GFlow.
        '''
        raise NotImplementedError("Subclasses should implement this method.")
