import numpy as np
import os
from os import listdir
from os.path import isfile, join
import collections
import re
from tqdm import trange
from tqdm import *
import random
#import pickle
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle
from utils import identifier_splitting
from tensorflow.keras.preprocessing.sequence import pad_sequences
import hashlib
import csv
import copy
import sys
from .base_tree_utils import BaseTreeUtils


class TreeLoader(BaseTreeUtils):
   
    def __init__(self, opt, training=True):
        super().__init__(opt)
        print("Loading existing data file: ", str(opt.data_path))
        self.all_subtrees_bucket, self.random_subtrees_bucket, self.bucket_sizes, self.trees = pickle.load(open(opt.data_path, "rb"))
        
        self.tree_size_threshold_lower = opt.tree_size_threshold_lower
        self.tree_size_threshold_upper = opt.tree_size_threshold_upper
        self.batch_size = opt.batch_size

        self.is_training = training
   

    def extract_training_data(self, tree_data):
        tree_path = tree_data["file_path"]
        tree, tokens, size, file_path = self.trees[tree_path]["tree"], self.trees[tree_path]["tokens"] , self.trees[tree_path]["size"], self.trees[tree_path]["file_path"]
        # print(tree)
        node_types = []
        node_tokens = []
        node_tokens_text = []
        node_indexes = []
        children_indices = []
        children_node_types = []
        children_node_tokens = []
        # label = 0

        # print("Label : " + str(label))
        queue = [(tree, -1)]
        # print queue
        while queue:
            # print "############"
            node, parent_ind = queue.pop(0)
            # print node
            # print parent_ind
            node_ind = len(node_types)
            # print "node ind : " + str(node_ind)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children_indices.append([])
            children_node_types.append([])
            children_node_tokens.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children_indices[parent_ind].append(node_ind)
                children_node_types[parent_ind].append(int(node["node_type"]))
                children_node_tokens[parent_ind].append(node["node_token"])
            
            node_type = node['node_type']
            node_token = node['node_token']
            node_token_text = node["node_token_text"]

            node_types.append(int(node_type))
            node_tokens.append(node_token)
            node_indexes.append(node_ind)
            node_tokens_text.append(node_token_text)


        token_ids = []
        token_ids.append(self.look_up_for_id_of_token("<GO>"))
        for token in tokens:
            token_ids.append(self.look_up_for_id_of_token(token))
        token_ids.append(self.look_up_for_id_of_token("<EOS>"))
     
        return node_indexes, node_types, node_tokens, node_tokens_text, children_indices, children_node_types, children_node_tokens, token_ids, size, file_path

                
    def make_batch(self, batch_data):
        batch_node_indexes = []
        batch_node_types = []
        batch_node_tokens = []
        batch_node_tokens_text = []
        batch_children_indices = []
        batch_children_node_types = []
        batch_children_node_tokens = []
        batch_tree_size = []
        batch_file_path = []
        batch_token_ids = []
        batch_length_targets = []
        batch_subtree_id = []
        for tree_data in batch_data:
            node_indexes, node_types, node_tokens, node_tokens_text, children_indices, children_node_types, children_node_tokens, token_ids, size, file_path = self.extract_training_data(tree_data)
            
            # random_subtree_id = self.random_sampling_subtree(tree_data["subtrees_ids"])
            batch_subtree_id.append(tree_data["subtree_id"])

            batch_node_indexes.append(node_indexes)
            batch_node_types.append(node_types)
            batch_node_tokens.append(node_tokens)
            batch_node_tokens_text.append(node_tokens_text)
            batch_children_indices.append(children_indices)
            batch_children_node_types.append(children_node_types)
            batch_children_node_tokens.append(children_node_tokens)
            batch_tree_size.append(size)
            batch_file_path.append(file_path)
            batch_token_ids.append(token_ids)
            batch_length_targets.append(len(token_ids))

        # print(batch_token_ids)
        batch_token_ids = pad_sequences(batch_token_ids, padding='post', value=self.look_up_for_id_of_token("<PAD>"))
        batch_node_types, batch_node_tokens, batch_children_indices, batch_children_node_types, batch_children_node_tokens = self._pad_batch(batch_node_types, batch_node_tokens, batch_children_indices, batch_children_node_types, batch_children_node_tokens)
        
        batch_obj = {
            "batch_node_indexes": batch_node_indexes,
            "batch_node_types": batch_node_types,
            "batch_node_tokens": batch_node_tokens,
            "batch_node_tokens_text": batch_node_tokens_text,
            "batch_children_indices": batch_children_indices,
            "batch_children_node_types": batch_children_node_types,
            "batch_children_node_tokens": batch_children_node_tokens,
            "batch_token_ids": batch_token_ids,
            "batch_length_targets": batch_length_targets,
            "batch_tree_size": batch_tree_size,
            "batch_file_path": batch_file_path,
            "batch_subtree_id": batch_subtree_id
        }
        return batch_obj

    def _pad_batch(self, batch_node_types, batch_node_tokens, batch_children_indices, batch_children_node_types, batch_children_node_tokens):
        # if not nodes:
            # return [], [], []
        # batch_node_types
        max_num_nodes = max([len(x) for x in batch_node_types])
        batch_node_types = [n + [0] * (max_num_nodes - len(n)) for n in batch_node_types]

        # batch_children_indices
        max_num_nodes = max([len(x) for x in batch_children_indices])
        max_num_children_per_node = max([len(c) for n in batch_children_indices for c in n])
        batch_children_indices = [n + ([[]] * (max_num_nodes - len(n))) for n in batch_children_indices]
        batch_children_indices = [[c + [0] * (max_num_children_per_node - len(c)) for c in sample] for sample in batch_children_indices]
        
        # batch_node_tokens
        max_num_nodes = max([len(x) for x in batch_node_tokens])
        max_num_children_per_node = max([len(c) for n in batch_node_tokens for c in n])
        batch_node_tokens = [n + ([[]] * (max_num_nodes - len(n))) for n in batch_node_tokens]
        batch_node_tokens = [[c + [0] * (max_num_children_per_node - len(c)) for c in sample] for sample in batch_node_tokens]

        # batch_children_node_types
        max_num_nodes = max([len(x) for x in batch_children_node_types])
        max_num_children_per_node = max([len(c) for n in batch_children_node_types for c in n])
        batch_children_node_types = [n + ([[]] * (max_num_nodes - len(n))) for n in batch_children_node_types]
        batch_children_node_types = [[c + [0] * (max_num_children_per_node - len(c)) for c in sample] for sample in batch_children_node_types]

        # batch_children_node_tokens
        # 0-dimension: number of nodes of the tree
        # 1-dimension: number of children per node
        # 2-dimension: number of subtoken per children per node
        max_num_nodes = max([len(x) for x in batch_children_node_tokens])
        max_num_children_per_node = max([len(c) for n in batch_children_node_tokens for c in n])
        max_num_of_subtoken_per_children_per_node = max([len(s) for n in batch_children_node_tokens for c in n for s in c])
        batch_children_node_tokens = [n + ([[]] * (max_num_nodes - len(n))) for n in batch_children_node_tokens]
        batch_children_node_tokens = [[c + [[]] * (max_num_children_per_node - len(c)) for c in sample] for sample in batch_children_node_tokens]  
        batch_children_node_tokens = [[[s + [0] * (max_num_of_subtoken_per_children_per_node - len(s)) for s in c] for c in sample] for sample in batch_children_node_tokens]
      
        return batch_node_types, batch_node_tokens, batch_children_indices, batch_children_node_types, batch_children_node_tokens

    def _produce_mask_vector(self, nodes):
        masks = []

        for n in nodes:        
            mask = [1 for i in range(len(n))]
            masks.append(mask)

        padded_inputs = pad_sequences(masks, padding='post')
        return padded_inputs

    def make_minibatch_iterator(self):
        buckets = self.random_subtrees_bucket

        # This part is important
        if not self.is_training:
            print("Using random subtrees buckets...........")
            buckets = self.random_subtrees_bucket
        else:
            print("Using all subtrees buckets...........")

        bucket_ids = list(buckets.keys())
        random.shuffle(bucket_ids)
        # for bucket_idx, bucket_data in buckets.items():
        for bucket_idx in bucket_ids:

            bucket_data = buckets[bucket_idx]
            print("Switching to bucket with size : " + str(self.trees[bucket_data[0]["file_path"]]["size"]))
            print("Number of items in bucket : " + str(len(bucket_data)))
            # print(file)
            # print("Shuffling data.....")
            random.shuffle(bucket_data)
            
            elements = []
            samples = 0

            # if self.is_training == True:
            #     sampling_size = int(len(bucket_data)*0.2)
            #     bucket_data = bucket_data[:sampling_size]
    
            for i, tree_data in enumerate(bucket_data):
                
                size = self.trees[tree_data["file_path"]]["size"]
                # if self.is_training:
                if size > self.tree_size_threshold_lower and size < self.tree_size_threshold_upper:
                    elements.append(tree_data)
                    samples += 1
                # else:
                #     elements.append(tree_data)
                #     samples += 1

                    
                if samples >= self.batch_size:
                    
                    batch_obj = self.make_batch(elements)
                    
                    batch_node_indicators = self._produce_mask_vector(batch_obj["batch_node_types"]) 
                    # for node in batch_nodes:
                    #     print(len(node))
                    batch = {}
                    batch["batch_node_indexes"] = batch_obj["batch_node_indexes"]
                    batch["batch_node_types"] = np.asarray(batch_obj["batch_node_types"])
                    batch["batch_node_tokens"] = np.asarray(batch_obj["batch_node_tokens"])
                    batch["batch_node_tokens_text"] = batch_obj["batch_node_tokens_text"]
                    batch["batch_children_indices"] = np.asarray(batch_obj["batch_children_indices"])
                    batch["batch_children_node_types"] = np.asarray(batch_obj["batch_children_node_types"])
                    batch["batch_children_node_tokens"] = np.asarray(batch_obj["batch_children_node_tokens"])
                    batch["batch_tree_size"] = batch_obj["batch_tree_size"]
                    batch["batch_file_path"] = batch_obj["batch_file_path"]
                    batch["batch_node_indicators"] = batch_node_indicators
                    batch["batch_token_ids"] = batch_obj["batch_token_ids"]
                    batch["batch_length_targets"] = batch_obj["batch_length_targets"]
                    batch["batch_subtree_id"] = np.reshape(batch_obj["batch_subtree_id"], (self.batch_size, 1))

            
                    yield batch
                    elements = []
                    samples = 0
