import pandas as pd
from src.utils.proxy_reward import proxy_reward_function
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from abc import ABC
import torch
class Word(ABC):
    '''
    We assume the sample space can be abstract into sentences composed of words. Then for word of cardinality k, sentence max length n
    the total number of sample is k^n
    This assumption about the sample gives us a structure between samples, and this class aims to recover the state of sample_structure
    given current collection of samples
    '''
    def __init__(self,args, evidences):
        self.evidences = evidences
        self.raw_samples = evidences['full_samples']
        self.evidence_data = evidences['evidence_data']
        elements = evidences['elements']
        if elements is not None:
            if 'end' not in elements:
                self.elements=['end',*elements]
            self.vocabulary = { element: index for index, element in enumerate(elements)}
        else:
            self.vocabulary = self.build_vocabulary(raw_data, embedding_dim , elements=None)
        self.embedding = nn.Embedding(len(elements),args.embedding_dim)
        self.proxy_reward = proxy_reward_function(args.scorer, evidences)
        #  self.env_structure = self.Word_to_Structure


    def tokenizer(self, sentence, vocab):
        return [vocab[word] for word in sentence if word in vocab]

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
        #    elements = somefunction(raw_data)
            raise Exception('The elements should be extracted from raw_data')
        return embedding

    def embed(self,tokenized_sentence_tensors):
        padded = pad_sequence(tokenized_sentence_tensors, batch_first=True, padding_value=0)
        return self.embedding(padded)

    def rawData_to_samples(self,scorer_name,format='unordered'):
        '''
        Convert the raw data to standard sample format.
        If the format is unordered, meaning the elements for composing an object has no ordering, we will use a trie to store
        the string representation of the raw data.
        :param format: The format determines the data structure used to store the raw data
        :return: (string representation of rawData, reward)
        '''
        if format =='unordered':
            # parallel computing to convert raw data to string representation with numpy
            tokenized_sentences = [ self.tokenizer(sentence,self.vocabulary) for sentence in self.raw_samples]
            tokenized_sentences_tensors = [tokens for tokens in tokenized_sentences]
            samples = tokenized_sentences_tensors
            #use proxy reward function to label each sample and put them into trie data structures
            rewards = self.proxy_reward.annotate(self.raw_samples)
        return {'object':samples,'rewards':rewards}