import math
import tensorflow as tf
import numpy as np
from bidict import bidict
from keras_radam.training import RAdamOptimizer
import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))

class BaseLayer():
    def __init__(self, opt):
        self.node_type_lookup = self.load_node_type_vocab(opt.node_type_vocabulary_path)
        self.node_token_lookup = self.load_node_type_vocab(opt.token_vocabulary_path)
        self.subtree_lookup = self.load_subtree_vocab(opt.subtree_vocabulary_path)

        self.batch_size = opt.batch_size

        self.placeholders = {}
        self.weights = {}
    
    def load_node_token_vocab(self, token_vocab_path):
        node_token_lookup = {}
        with open(token_vocab_path, "r") as f:
            data = f.readlines()
           
            for i, line in enumerate(data):
                line = line.replace("\n", "").strip()
                node_token_lookup[line] = i

        return bidict(node_token_lookup)

    def load_node_type_vocab(self, node_type_vocab_path):
        node_type_lookup = {}
        with open(node_type_vocab_path, "r") as f:
            data = f.readlines()
           
            for i, line in enumerate(data):
                line = line.replace("\n", "").strip()
                node_type_lookup[line.upper()] = i

        return bidict(node_type_lookup)


    def load_subtree_vocab(self, subtree_vocab_path):
        subtree_lookup = {}
        with open(subtree_vocab_path, "r") as f:
            data = f.readlines()
           
            for i, line in enumerate(data):
                line = line.replace("\n", "").strip()
                subtree_lookup[line.upper()] = i

        return bidict(subtree_lookup)