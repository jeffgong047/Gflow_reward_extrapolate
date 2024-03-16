from abc import ABC
from gfn.states import States
from gfn.actions import Actions
import torch
from torch.nn.utils.rnn import pad_sequence
class States_triv(States):
    def __init__(self,tensor):
        self.tensor = tensor




class translator(ABC):
    '''
    The goal of this object is to ensure efficient communication between Trie implementation of the gflow++ with torchgfn package
    through unified states and action representation offered by torchgfn
    we found that the translator approach is too tedious, the most efficient way is to let the environment directly compatible with gflow++ due to
    the simplicity of data types. This translator might be useful for some cases that the api call is not too often.
    '''
    def __init__(self):
        self.languages = ['torchgfn','gflow++']


    def translate(self,data, gflow_plus_plus=None, env= None,):
        if isinstance(data ,States):
            if len(States.shape) ==2 :
                return States.tensor.tolist()
        elif isinstance(data, Actions):
            pass
        elif isinstance(data, list):
            if len(data)==0:
                return env.States(env.s0)
            # if data[-1] == gflow_plus_plus.vocabulary['end']: #Probably need to consider end case later
            #     return env.sf
            if isinstance(data[0], list):
                batch_data = [torch.tensor(d) for d in data]
                padded_batch = pad_sequence(batch_data, batch_first = True)
                padded_batch[padded_batch == 0 ] = -1
                tensor_data = padded_batch
            else:
                tensor_data = torch.tensor(data)
            return States_triv(tensor_data)



