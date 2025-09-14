# Node2vec & DeepWalk Implementation from Scratch

This repository contains a from-scratch implementation of Node2vec, a scalable feature learning algorithm for graphs. Node2vec learns continuous vector representations for nodes using biased random walks and the skip-gram model, enabling downstream tasks such as classification, link prediction, and clustering. This implementation allows for **flexible context window** as well as flexibility in specifying the **distribution** used to generate **negative samples**.

## Repository Structure
- `example_gen_embeddings.ipynb` is an **example** Jupyter Notebook showing how to **generate node embeddings** using the Python scripts in this repository. This notebook generates a graph from the Stochastic Block Model and then generates node embeddings for that graph.
- `node_classification_pubmed_data.ipynb` contains code for **node classification on the PubMed citation network data set**.
- `alias_sampler.py` implements alias sampling to sample an integer between $1$ to $n$ in $O(1)$ time.
- `gen_node2vec_walks.py` implements node2vec/DeepWalk walk generation using alias sampling for computational efficiency.
- `helper_classes.py` implements a PyTorch class for the dataset of co-occurences created using the random walks. It also implements Word2vec which is the underlying algorithm used to generate node embeddings in both node2vec and DeepWalk.
- `gen_embeddings.py` contains functions to train node embeddings. The function `gen_embeddings` in this file takes walks, among other inputs, and trains node embeddings. The other functions in this file implements logging (for debugging) and for saving checkpoints.

## Requirements
- torch
- networkx
- numpy
- scikit-learn