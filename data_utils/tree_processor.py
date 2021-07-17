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
import hashlib
import csv
import copy
from bidict import bidict
from .base_tree_utils import BaseTreeUtils
import sys
csv.field_size_limit(sys.maxsize)

class TreeProcessor(BaseTreeUtils):
   
    def __init__(self, opt):
        super().__init__(opt)
        tree_directory = opt.input_data_directory
        subtree_directory = opt.subtree_directory
        all_subtrees_path = opt.subtree_vocabulary_path
        self.process_data_for_training = opt.training
        if self.process_data_for_training == 1:
            print("Processing data for training!!")
        else:
            print("Processing data for inferring!!")
        self.output_path = opt.output_path
        # if os.path.exists(saved_input_filename):
        #     print("Loading existing data file: ", str(saved_input_filename))
        #     self.train_buckets, self.val_buckets, self.bucket_sizes, self.trees = pickle.load(open(saved_input_filename, "rb"))
           
        # else:
        self.trees = self.load_program_data(tree_directory, subtree_directory)
        

    def process_data(self):
        self.train_buckets, self.val_buckets, self.bucket_sizes = self.put_trees_into_bucket(self.trees)
        print("Serializing......")
        self.data = (self.train_buckets, self.val_buckets, self.bucket_sizes, self.trees)
        pickle.dump(self.data, open(self.output_path, "wb" ) )

    def load_program_data(self, tree_directory, subtree_vocab_directory):
        # trees is to store meta data of trees
        trees = {}
        # trees_dict is to store dictionay of tree, key is is file path, content is the tree 
        # trees_dict = {}
        all_subtrees_dict = {}

        # This condition is to check if we need to load the subtrees or not
        # For training, we need to load the subtrees. For testing, we dont need
        if self.process_data_for_training:
            for subdir , dirs, files in os.walk(subtree_vocab_directory): 
                for file in tqdm(files):
                    subtree_file_path = os.path.join(subdir,file)
                    subtree_file_name = os.path.basename(subtree_file_path)
                    subtree_file_name = subtree_file_name.replace(".ids.csv", "")
                    all_subtrees_dict[subtree_file_name] = subtree_file_path

        for subdir , dirs, files in os.walk(tree_directory): 
            for file in tqdm(files):
                
                if file.endswith(".pkl") and not file.endswith(".slice.pkl"):

                    pkl_file_path = os.path.join(subdir,file)
                
                    file_name = os.path.basename(pkl_file_path).replace(".pkl", "")
                    
                    srcml_representation = self.load_tree_from_pickle_file(pkl_file_path)
                    # print(pb_representation)
                    root = srcml_representation.element

                    tree, size, tokens = self._traverse_tree(root)
                    tree_data = {
                        "tree": tree,
                        "tokens": tokens,
                        "size": size,
                        "file_path": pkl_file_path
                    }

                    if self.process_data_for_training == 1:
                        if file_name in all_subtrees_dict:
                            subtree_file_path = all_subtrees_dict[file_name]
                            print(subtree_file_path)

                            if os.path.exists(subtree_file_path):
                                print("Loading subtrees from : ", subtree_file_path)
                                file_subtrees_dict, subtrees_ids = self.load_subtrees(subtree_file_path)
                                # print(file_subtrees_dict)
                                if len(subtrees_ids) > 0:

                                    tree_data["subtrees_dict"] = file_subtrees_dict
                                    tree_data["subtrees_ids"] = subtrees_ids
                                    
                                    
                        else:
                            print("Missing subtrees : ", file_name)
                    trees[pkl_file_path] = tree_data
                          
        return trees

 
    def _traverse_tree(root, text):
        num_nodes = 0
        root_type = str(root.type)
      
        queue = [root]

        root_json = {
            "node_type": root_type,
            "node_token": [],
            "node_token_text": "",
            "children": []
        }

        queue_json = [root_json]
        while queue:
            
            current_node = queue.pop(0)
            current_node_json = queue_json.pop(0)
            num_nodes += 1

            children = [x for x in current_node.children]
            queue.extend(children)

            for child in children:
                child_type = str(child.type)
              
                child_token = ""
                children_sub_token_ids = []
                child_sub_tokens = []
                if not has_child(child):
                    child_token = str(text[child.start_byte:child.end_byte])
                    child_sub_tokens = identifiersplitting.split_identifier_into_parts(child_token)
                    child_sub_tokens = remove_noisy_tokens(child_sub_tokens)

                    for sub_token in child_sub_tokens:
                        sub_token = process_token(sub_token)
                        children_sub_token_ids.append(self.look_up_for_id_of_token(sub_token))


                if len(children_sub_token_ids) == 0:
                    children_sub_token_ids.append(0)
               
                # print(children_sub_token_ids)
                child_json = {
                    "node_type": child_type,
                    "node_token": children_sub_token_ids,
                    "node_token_text":child_token,
                    "children": []
                }

                current_node_json['children'].append(child_json)
                queue_json.append(child_json)

        return root_json, num_nodes
    
    # Not really put the trees into buckets, only put the tree_path and the sub_id into buckets to reduce the cost of computation
    def put_trees_into_bucket(self, trees):
        bucket_sizes = np.array(list(range(30 , 7500 , 10)))

        """
        Maintain two copy of the buckets of the same dataset for different purposes
        """
        all_subtrees_bucket = defaultdict(list)
        random_subtrees_bucket = defaultdict(list)

        for tree_path, tree_data in trees.items():

            tree_size = tree_data["size"]
            chosen_bucket_idx = np.argmax(bucket_sizes > tree_size)
            
            if self.process_data_for_training == 1:
                print("Putting these ids to bucket : " + str(tree_data["subtrees_ids"]))
                for subtree_id in tree_data["subtrees_ids"]:
                    temp_bucket_data = {}
                    temp_bucket_data["file_path"] = tree_path
                    temp_bucket_data["subtree_id"] = subtree_id
                    
                    all_subtrees_bucket[chosen_bucket_idx].append(temp_bucket_data)

                random_subtrees_bucket[chosen_bucket_idx].append(random.choice(all_subtrees_bucket[chosen_bucket_idx]))
            else:
                random_subtrees_bucket[chosen_bucket_idx].append(tree_data)

        return all_subtrees_bucket, random_subtrees_bucket, bucket_sizes
