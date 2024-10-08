import jax.numpy as jnp
from abc import ABC, abstractmethod
import ast
import marisa_trie
class sample(ABC):
    def __init__(self, array_like):
        self.sample_seq = jnp.array(array_like)
        self.len = len(array_like)
    def back_track(self, back_steps):
        back_steps +=1
        return self.sample_seq[self.len-back_steps],




class samples(sample):
    '''
    samples class was written in the spirit to organize the training data better for running gflow-extrapolate algorithms
    Our implementation of samples are based upon strings
    '''
    def __init__(self,array_like):
        self.samples_seq = jpn.array(array_like)
        self.num_samples = array_like[0]
        self.samples_structure = self.organize_samples(self.samples_seq)
    def organize_samples(self,samples):
        samples_structure = marisa_trie.Trie(samples)
        return samples_structure

    def samplesWithPrefix(self, prefix):
        return self.samples_structure.keys(prefix)
    def samplesPrefixQuery(self, query):
        return self.samples_structure.prefixes(query)



    # def __setitem__(self, key, value):
    #     head = key[0]
    #     if head in self.path:
    #         node = self.path[head]
    #     else:
    #         node = Trie()
    #         self.path[head] = node
    #
    #     if len(key) > 1:
    #         remains = key[1:]
    #         node.__setitem__(remains, value)
    #     else:
    #         node.value = value
    #         node.value_valid = True
    #
    # def __delitem__(self, key):
    #     head = key[0]
    #     if head in self.path:
    #         node = self.path[head]
    #         if len(key) > 1:
    #             remains = key[1:]
    #             node.__delitem__(remains)
    #         else:
    #             node.value_valid = False
    #             node.value = None
    #         if len(node) == 0:
    #             del self.path[head]
    #
    # def __getitem__(self, key):
    #     head = key[0]
    #     if head in self.path:
    #         node = self.path[head]
    #     else:
    #         raise KeyError(key)
    #     if len(key) > 1:
    #         remains = key[1:]
    #         try:
    #             return node.__getitem__(remains)
    #         except KeyError:
    #             raise KeyError(key)
    #     elif node.value_valid:
    #         return node.value
    #     else:
    #         raise KeyError(key)
    #
    # def __contains__(self, key):
    #     try:
    #         self.__getitem__(key)
    #     except KeyError:
    #         return False
    #     return True
    #
    # def __len__(self):
    #     n = 1 if self.value_valid else 0
    #     for k in self.path.keys():
    #         n = n + len(self.path[k])
    #     return n
    #
    # def get(self, key, default=None):
    #     try:
    #         return self.__getitem__(key)
    #     except KeyError:
    #         return default
    #
    # def nodeCount(self):
    #     n = 0
    #     for k in self.path.keys():
    #         n = n + 1 + self.path[k].nodeCount()
    #     return n
    #
    # def keys(self, prefix=[]):
    #     return self.__keys__(prefix)
    #
    # def __keys__(self, prefix=[], seen=[]):
    #     result = []
    #     if self.value_valid:
    #         isStr = True
    #         val = ""
    #         for k in seen:
    #             if type(k) != str or len(k) > 2:
    #                 isStr = False
    #                 break
    #             else:
    #                 val += k
    #         if isStr:
    #             result.append(val)
    #         else:
    #             result.append(prefix)
    #     if len(prefix) > 0:
    #         head = prefix[0]
    #         prefix = prefix[1:]
    #         if head in self.path:
    #             nextpaths = [head]
    #         else:
    #             nextpaths = []
    #     else:
    #         nextpaths = self.path.keys()
    #     for k in nextpaths:
    #         nextseen = []
    #         nextseen.extend(seen)
    #         nextseen.append(k)
    #         result.extend(self.path[k].__keys__(prefix, nextseen))
    #     return result
    #
    # def __iter__(self):
    #     for k in self.keys():
    #         yield k
    #     raise StopIteration
    #
    # def __add__(self, other):
    #     result = Trie()
    #     result += self
    #     result += other
    #     return result
    #
    # def __sub__(self, other):
    #     result = Trie()
    #     result += self
    #     result -= other
    #     return result
    #
    # def __iadd__(self, other):
    #     for k in other:
    #         self[k] = other[k]
    #     return self
    #
    # def __isub__(self, other):
    #     for k in other:
    #         del self[k]
    #     return self