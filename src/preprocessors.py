import pandas as pd
import utils.proxy_reward as proxy_reward
from utils.data_structures import samples
import torch.nn as nn
class Word(ABC):
    '''
    We assume the sample space can be abstract into sentences composed of words. Then for word of cardinality k, sentence max length n
    the total number of sample is k^n
    This assumption about the sample gives us a structure between samples, and this class aims to recover the state of sample_structure
    given current collection of samples
    '''
    def __init__(self,args,  raw_data,elements):
        self.element_to_index = { element: index for index, element in enumerate(elements)}
        self.vocabulary = self.build_vocabulary(raw_data, embedding_dim , elements=None)
        self.proxy_reward = proxy_reward(args.proxy_reward)
        self._data = self.rawData_to_samples(args.format, raw_data)
        #  self.env_structure = self.Word_to_Structure



    def build_vocabulary(self,raw_data, embedding_dim,elements):
        '''
        we build a vocabulary to abstract all the possible elements used for building the object.
        First build an index for the elements then map them to vectors.
        :input: raw_data: a
                elements: indexed list of elements
        :return: vector representation of each elements where we can easily look up through its index
        '''
        # Transform raw data into structured format
        # Let's assume structured format is a dictionary of word frequencies
        if not elements:
            #extract all elements from the raw_data
            elements = somefunction(raw_data)
            raise Exception('The elements should be extracted from raw_data')

        embedding = nn.Embedding(len(elements),embedding_dim)
        return embedding

    def rawData_to_samples(self, raw_data,format='unordered'):
        '''
        Convert the raw data to standard sample format.
        If the format is unordered, meaning the elements for composing an object has no ordering, we will use a trie to store
        the string representation of the raw data.
        :param format: The format determines the data structure used to store the raw data
        :return: (string representation of rawData, reward)
        '''
        if format =='unordered':
            # parallel computing to convert raw data to string representation with numpy
            idx = vocab[raw_data]
            idx_tensor = torch.tensor([idx])
            samples = self.vocabulary(idx_tensor)
            #use proxy reward function to label each sample and put them into trie data structures
            rewards = self.proxy_reward(raw_data)

        return (samples,rewards)


