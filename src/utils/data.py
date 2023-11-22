import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from pgmpy.metrics import structure_score
from pgmpy.base import DAG



def get_data(args):
    name = args.objects
    if name =="demo":
        edges = [('A', 'B'), ('C', 'B'), ('C', 'D')]
        filename ='data/demo.csv'
        evidence_data = pd.read_csv(filename, delimiter=',', dtype='category')
        score ='Bdeu'
    elif name =="demo1":
        edges = [('A','B'),('B','C')]
        graph = nx.DiGraph(edges)
        filename ='data/demo1.csv'
        evidence_data = pd.read_csv(filename, delimiter=',', dtype='category')
        score ='BDeu'
    elif name == "demo2":
        score = 'BDeu'
        evidence_data = pd.read_csv('/common/home/hg343/Research/Gflow_reward_extrapolate/data/demo2.csv')
        nodes = ['A','B','C']
        elements = []
        for u_i,u in enumerate(nodes):
            for v_i, v in enumerate(nodes):
                if u_i!=v_i:
                    elements.append(u+v)
        full_samples =list(itertools.chain.from_iterable([itertools.combinations(elements,i+1) for i in range(len(elements))]))
    elif name =="demo3":
        edges = [('B','A'),('B','C')]
        graph = nx.DiGraph(edges)
        filename ='data/demo3.csv'
        evidence_data = pd.read_csv(filename, delimiter=',', dtype='category')
        score ='BDeu'
    elif name =='demo4':
        edges = [('A', 'B'), ('C', 'B'), ('C', 'D'),('E','C'),('C','A')]
        graph = BayesianNetwork(edges)
        filename = args.data_path
        evidence_data = pd.read_csv(filename, delimiter=',', dtype='category')
        score = 'BDeu'
    return score,evidence_data ,elements,full_samples