import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Relu, Serial

class EdgeFlowPredictor:
    def __init__(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def fit(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, inputs):
        '''
        For edges which belongs to some samples, we can better estimates its edge flow
        :param inputs:
        :return:
        '''
        raise NotImplementedError("Subclasses should implement this method.")

    def extrapolate(self, inputs):
        '''
        For edges that belongs to non of the samples, we extrapolate its edge flow
        :param inputs:
        :return:
        '''
        # Extrapolation logic here
        raise NotImplementedError("Subclasses should implement this method.")

