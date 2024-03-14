import sklearn
import numpy as np

def l1_distance(pi,p):
    '''

    :param pi policy_distribution:
    :param p ground_truth_distribution:
    :return:  E[|p(x)-pi(x)|]
    '''
    pi = np.array(pi)
    p = np.array(p)
    return np.sum(np.abs(pi-p)*p)



