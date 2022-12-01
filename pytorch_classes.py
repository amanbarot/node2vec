class word2vec(nn.Module):
    """PyTorch implementation of Word2vec
    """
    def __init__(self, vocab_size, embedding_dim, word_dist, neg_count, device):
        """
        Input:
        ------
        vocab_size: total number of words in the vocabulary. In the 
            context of graphs, this is the number of nodes
        embedding_dim: The dimension to embed the words in
        word_dist: The marginal distribution of words. This is used to generate
            negative samples.
        neg_count: The number of negative samples to generate for each 
            word-context pair in the data set.
        """
        super(word2vec, self).__init__()
        self.inp_emb = nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=embedding_dim)
        self.out_emb = nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=embedding_dim)
        self.word_dist = torch.tensor(word_dist, dtype=torch.float64, device=device)
        self.neg_count = torch.tensor(neg_count, device=device)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inp, out):
        """
        Input:
        ------
        inp: The input context word
        out: The output center word
        """
        x = torch.sum(torch.log(self.sigmoid(torch.bmm(self.inp_emb(inp).unsqueeze(1),
                                                       self.out_emb(out).unsqueeze(2)))))
        neg_samples = torch.multinomial(self.word_dist, 
                                        self.neg_count * inp.shape[0], 
                                        replacement=True)
        x = x + torch.sum(torch.log(self.sigmoid(-1 * torch.bmm(\
                  torch.cat([self.inp_emb(inp) for x in range(self.neg_count)],
                            0).unsqueeze(1),
                  self.out_emb(neg_samples).unsqueeze(2)))))
        return(-1 * x)


class cooccurenceDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset object for co-occurences computed from random walks
    """
    def __init__(self, walks, t_L, t_U):
        """An instantiation method. It converts the list of walks to a list of edge pairs
        representing the co-occurences.
        Input:
        ------
        walks (list): A list of random walks on the graph
        t_L (int): Lower limit of the window for computing co-occurences
        t_U (int): Upper limit of the window for computing co-occurences
        """
        self.co_occL = []
        assert t_L <= t_U
        for walk in walks:
            for t in range(t_L, t_U + 1):
                for k in range(len(walk) - t):
                    self.co_occL.append((int(walk[k]), int(walk[k + t])))
                    self.co_occL.append((int(walk[k + t]), int(walk[k])))
    
    def __len__(self):
        """Returns the length of the list of co-occurences
        """
        return len(self.co_occL)
    
    def __getitem__(self, idx):
        """Returns the co-occurence at index idx
        """
        return self.co_occL[idx]
