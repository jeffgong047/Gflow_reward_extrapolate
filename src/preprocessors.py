import pandas as pd
import utils.proxy_reward as pr
from utils.data_structures import samples
class Word():
    '''
    We assume the sample space can be abstract into sentences composed of words. Then for word of cardinality k, sentence max length n
    the total number of sample is k^n
    This assumption about the sample gives us a structure between samples, and this class aims to recover the state of sample_structure
    given current collection of samples
    '''
    def __init__(self,args,  raw_data):
        self.path = {}
        self.value = None
        self.value_valid = False
        self.env_structure = self.Word_to_Structure
        self.samples = self.rawData_to_samples(args.format)
        self._data = raw_data
        self.proxy_reward = pr.args.proxy_reward

    def vocabulary(self):
        '''
        we build a vocabulary to abstract all the possible elements used for building the object.
        First build an index for the elements then map them to vectors.
        :return: vector representation of each elements where we can easily look up the element representation through its index
        '''
        # Transform raw data into structured format
        # Let's assume structured format is a dictionary of word frequencies
        raw_data = self.rawData()
        word_freq = {}
        for sentence in raw_data:
            for word in sentence.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        return word_freq

    def rawData_to_samples(self, format='unordered'):
        '''
        Convert the raw data to standard sample format.
        If the format is unordered, meaning the elements for composing an object has no ordering, we will use a trie to store
        the string representation of the raw data.
        :param format: The format determines the data structure used to store the raw data
        :return:
        '''
        if format =='unordered':
            # parallel computing to convert raw data to string representation with numpy
            pass
            samples_with_word = [sentence for sentence in raw_data if Word in sentence]
            #use proxy reward function to label each sample and put them into trie data structures
            pass
        return samples


