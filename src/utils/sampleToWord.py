from ABC import abc
import jax.numpy as jnp


class Word():
    '''
    We assume the sample space can be abstract into sentences composed of words. Then for word of cardinality k, sentence max length n
    the total number of sample is k^n
    This assumption about the sample gives us a structure between samples, and this class aims to recover the state of sample_structure
    given curremt collection of samples
    '''
    def __init__(self,Word,sapmles):
        self.path = {}
        self.value = None
        self.value_valid = False
        self.env_structure = self.Word_to_Structure

    def rawData(self):
        pass
    def language_from_rawData(self):
        pass
    def rawData_to_samples(self,Word):
        pass



