import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import bisect

class cooccurenceDataset(Dataset):
    """
    A PyTorch Dataset that samples co-occurrences from random walks.
    It does not load all co-occurrences into memory, but computes
    them on-the-fly by decoding a flattened list of co-occurrences
    from the walks.

    Assumptions:
    - Walks are of the same length. This will happen if the graph
        does not have isolated nodes.
    - The walks are long enough to allow for the specified t_L and t_U.
    """
    def __init__(self, walks, t_L=1, t_U=5):
        """
        Args:
            walks (list of list of int): List of random walks, 
                where each walk is a list of node indices.
            t_L (int): Minimum distance between center and context nodes.
            t_U (int): Maximum distance between center and context nodes.
        """
        self.walks = walks
        self.t_L = t_L
        self.t_U = t_U
        self.L = len(walks[0])
        if self.L <= self.t_U:
            raise ValueError(
                f"Walks must be longer than t_U={t_U}. \
                    Got walk length {self.L}."
                    )
        num_coocc_per_walk = 2 * (self.L - self.t_U) \
            * (self.t_U - self.t_L + 1)
        
        # self.walk_offsets contains starting index of each walk 
        # in the flattened dataset of cooccurrences.
        self.walk_offsets = [0] 
        for _ in walks:
            self.walk_offsets.append(
                self.walk_offsets[-1] + num_coocc_per_walk
                )

    def __len__(self):
        """
        Returns:
            int: Total number of co-occurrence pairs in the dataset.
        """
        # The last element in walk_offsets is the total number of pairs
        # across all walks.
        return self.walk_offsets[-1]

    def __getitem__(self, idx):
        # Find which walk this idx belongs to
        walk_idx = 0
        walk_idx = bisect.bisect_right(self.walk_offsets, idx) - 1
        local_idx = idx - self.walk_offsets[walk_idx]
        walk = self.walks[walk_idx]

        # Decode center-context from flat index
        base = 0
        for t in range(self.t_L, self.t_U + 1):
            num_pairs = 2 * (self.L - t)
            if local_idx < base + num_pairs:
                k = (local_idx - base) // 2
                direction = (local_idx - base) % 2
                if direction == 0:
                    return walk[k], walk[k + t]
                else:
                    return walk[k + t], walk[k]
            base += num_pairs

        raise IndexError("Invalid index decoding.")

def cooc_collate(batch):
    """
    batch: list of (center, context) pairs
    Returns: 2 tensors of shape [batch_size]
    """
    centers, contexts = zip(*batch)
    return (
        torch.tensor(centers, dtype=torch.long), 
        torch.tensor(contexts, dtype=torch.long)
        )


class word2vec(nn.Module):
    """PyTorch implementation of Word2vec
    """
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 word_dist, 
                 neg_count, 
                 device):
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

        self.inp_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            dtype=torch.float32,
            )
        self.out_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            dtype=torch.float32,
            )
        self.word_dist = torch.tensor(
            word_dist, 
            dtype=torch.float32, 
            device=device
            )
        self.neg_count = neg_count
    
    def forward(self, inp, out):
        """
        Input:
        ------
        inp: The input context word
        out: The output center word
        """
        pos_score = torch.sum(
            self.inp_emb(inp) * self.out_emb(out), dim=1
        ) # B
        pos_loss = torch.sum(F.logsigmoid(pos_score + 1e-10))

        neg_samples = torch.multinomial(
            self.word_dist, 
            self.neg_count * inp.shape[0], 
            replacement=True
            ).view(inp.shape[0], self.neg_count) # B x K ids of negative samples
        neg_inp = self.inp_emb(inp).unsqueeze(2) # B x D x 1
        neg_out = self.out_emb(neg_samples) # B x K x D
        neg_score = torch.bmm(neg_out, neg_inp).squeeze(2) # B x K
        neg_loss = torch.sum(F.logsigmoid(-1 * neg_score + 1e-10))

        return -1 * (pos_loss + neg_loss)
