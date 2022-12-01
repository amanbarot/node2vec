# node2vec
The repository contains two classes for coding node2vec:
1. The class `word2vec` is for coding the algorithm word2vec. 
2. The class `cooccurenceDataset` takes a list of random walks and creates a list of co-occurences. In particular, unlike standard implementations this class allows for varying the parameters $t_L$ and $t_U$ to change range of the context. The dataset of co-occurences can then be used to compute node embeddings by using an instance of the word2vec class.
