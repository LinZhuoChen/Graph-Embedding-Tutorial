from gensim.models import Word2Vec
import pandas as pd
from joblib import Parallel, delayed
import itertools
import random
import networkx as nx
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

class DeepWalk:
    def __init__(self, G, num_walks, walk_length, workers=1):
        self.G = G 
        self.w2v_model = None
        self._embeddings = {}

        self.sentences = self.random_walk(num_walks=num_walks, walk_length=walk_length, 
                                            parallel_workers=workers, verbose=1)
    def train(self, embedding_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embedding_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done")

        self.w2v_model = model
        return model
    
    def get_embeddings(self):
        if self.w2v_model is None:
            print("model has not trained")
            return {}
        self._embeddings = {}
        for word in self.G.nodes():
            self._embedding[word] = self.w2v_model.wv[word]
        
        return self._embeddings

    def _single_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            # 依据G的邻域随机游走
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk
    def _multi_walks(self, nodes, num_walks, walk_length):
        walks = []
        for i in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self._single_walk(walk_length=walk_length, start_node=v))
        return walks
    def random_walk(self, num_walks, walk_length, parallel_workers=1, verbose=0):
        nodes = list(self.G.nodes())

        results = Parallel(n_jobs=parallel_workers, verbose=verbose)(delayed(self._multi_walks)(nodes, num, walk_length)
        for num in partition_num(num_walks, parallel_workers))

        return list(itertools.chain(*results))
    
def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]


df = pd.read_csv("space_data.tsv", sep = '\t')
G = nx.from_pandas_edgelist(df, "source", "target", edge_attr=True, create_using=nx.graph())
deepwalker = DeepWalk(G, num_walks=5, walk_length=10)
deepwalker.train(window_size=4, iter=3)
embeddings = deepwalker.get_embeddings()

terms = ['lunar escape systems','soviet moonshot', 'soyuz 7k-l1', 'moon landing',
         'space food', 'food systems on space exploration missions', 'meal, ready-to-eat',
         'space law', 'metalaw', 'moon treaty', 'legal aspects of computing',
         'astronaut training', 'reduced-gravity aircraft', 'space adaptation syndrome', 'micro-g environment']

embeddings_list = [embeddings[term] for term in terms]

embeddings_2d = PCA(n_components=2).fit_transform(embeddings_list)

plt.figure(figsize=(12, 9))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
for i, word in enumerate(terms):
    plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.show()







