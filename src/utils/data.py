import proxy_reward


def get_data(args):
    if name =="demo":
        edges = [('A', 'B'), ('C', 'B'), ('C', 'D')]
        graph =  nx.DiGraph(edges)
        filename ='data/demo.csv'
        data = pd.read_csv(filename, delimiter=',', dtype='category')
        proxy_reward = proxy_reward.bdeu_score
        score ='bde'
    elif name =="demo1":
        edges = [('A','B'),('B','C')]
        graph = nx.DiGraph(edges)
        filename ='data/demo1.csv'
        data = pd.read_csv(filename, delimiter=',', dtype='category')
        proxy_reward = proxy_reward.bdeu_score
        score ='bde'
    elif name == "demo2":
        edges = [('A','B'),('C','B')]
        graph = nx.DiGraph(edges)
        filename ='data/demo2.csv'
        data = pd.read_csv(filename, delimiter=',', dtype='category')
        proxy_reward = proxy_reward.bdeu_score
        score ='bde'
    elif name =="demo3":
        edges = [('B','A'),('B','C')]
        graph = nx.DiGraph(edges)
        filename ='data/demo3.csv'
        data = pd.read_csv(filename, delimiter=',', dtype='category')
        proxy_reward = proxy_reward.bdeu_score
        score ='bde'