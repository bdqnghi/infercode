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

excluded_tokens = [",","{",";","}",")","(",'"',"'","`",""," ","[]","[","]","/",":",".","''","'.'","b", "\\", "'['", "']","''"]

def _onehot(i, total):
    zeros = np.zeros(total)
    zeros[i] = 1.0
    return zeros

def _soft_onehot(list_i, total):
    zeros = np.zeros(total)
    for i in list_i:
        zeros[i] = 1
    return zeros

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def process_token(token):
    for t in excluded_tokens:
        token = token.replace(t, "")
        token = re.sub(r'[^\w]', ' ', token)
    return token

def remove_noisy_tokens(tokens):
    temp_tokens = []
    for t in tokens:
        if t not in excluded_tokens:
            t = process_token(t)
            if t:
                temp_tokens.append(t)
    return temp_tokens


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


class TreeProcessor(BaseTreeUtils):
   
    def __init__(self, opt):
        super().__init__(opt)
        tree_directory = opt.data_directory
        subtree_directory = opt.subtree_directory
        all_subtrees_path = opt.subtree_vocabulary_path
   

        base_name =os.path.basename(tree_directory)
        parent_base_name = os.path.basename(os.path.dirname(tree_directory))
        base_path = str(os.path.dirname(tree_directory))
        self.saved_input_filename = "%s/%s-%s.pkl" % (base_path, parent_base_name, base_name)

        # if os.path.exists(saved_input_filename):
        #     print("Loading existing data file: ", str(saved_input_filename))
        #     self.train_buckets, self.val_buckets, self.bucket_sizes, self.trees = pickle.load(open(saved_input_filename, "rb"))
           
        # else:
        self.trees = self.load_program_data(tree_directory, subtree_directory)
        

    def process_data(self):
        self.train_buckets, self.val_buckets, self.bucket_sizes = self.put_trees_into_bucket(self.trees)
        print("Serializing......")
        self.data = (self.train_buckets, self.val_buckets, self.bucket_sizes, self.trees)
        pickle.dump(self.data, open(self.saved_input_filename, "wb" ) )


    def load_subtrees(self, subtree_file_path):

        file_subtrees_dict = {}

        with open(subtree_file_path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter= ",")
            for row in reader:
                # print("-------------")
                root_node = row[0]
                subtree_nodes = row[1]
                depth = row[2]

                root_id = root_node.split("-")[0]
                root_type = root_node.split("-")[1]
               
                subtree_nodes = subtree_nodes.split(" ")

                # print(root)
                # print(subtrees)
                single_subtree = [root_type]
                for subtree_node in subtree_nodes:
                    # print(subtree)
                    if len(subtree_node.split("-")) == 3:
                        subtree_node_id = subtree_node.split("-")[0]
                        subtree_node_type = subtree_node.split("-")[1]
                        single_subtree.append(subtree_node_type)

                if len(single_subtree) >= 2:
                    str_features = "_".join(single_subtree)
                    hash_object = hashlib.md5(str_features.encode()).hexdigest()

                    file_subtrees_dict[hash_object] = str_features

                # print(whole_subtree_node_types)

        subtree_ids = []
        for k in file_subtrees_dict.keys():
            if k in self.subtree_lookup:
                subtree_ids.append(self.subtree_lookup[k])


        return file_subtrees_dict, subtree_ids




    def load_program_data(self, tree_directory, subtree_vocab_directory):
        # trees is to store meta data of trees
        trees = {}
        # trees_dict is to store dictionay of tree, key is is file path, content is the tree 
        # trees_dict = {}
        all_subtrees_dict = {}

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
                    
                    if file_name in all_subtrees_dict:
                        subtree_file_path = all_subtrees_dict[file_name]
                        print(subtree_file_path)

                        if os.path.exists(subtree_file_path):
                            print("Loading subtrees from : ", subtree_file_path)
                            file_subtrees_dict, subtrees_ids = self.load_subtrees(subtree_file_path)
                            # print(file_subtrees_dict)
                            if len(subtrees_ids) > 0:

                                # label = int(pkl_file_path_splits[len(pkl_file_path_splits)-2]) - 1 # uncomment this line later if there are bugs
                                # print(pkl_file_path)
                                pb_representation = self.load_tree_from_pickle_file(pkl_file_path)
                                # print(pb_representation)
                                root = pb_representation.element

                                tree, size, tokens = self._traverse_tree(root)
                                
                                # print(tokens)
                                tree_data = {
                                    "tree": tree,
                                    "tokens": tokens,
                                    "subtrees_dict": file_subtrees_dict,
                                    "subtrees_ids": subtrees_ids,
                                    "size": size,
                                    "file_path": pkl_file_path
                                }

                                trees[pkl_file_path] = tree_data
                                # trees.append(tree_data)
                    else:
                        print("Missing subtrees : ", file_name)
                          
        return trees

 


    def _traverse_tree(self, root):
        num_nodes = 0

        queue = [root]

        root_token = str(root.text)
        root_sub_tokens = identifier_splitting.split_identifier_into_parts(root_token)
        root_sub_tokens = remove_noisy_tokens(root_sub_tokens)

        tree_tokens = []
        tree_tokens.extend(root_sub_tokens)

        root_sub_token_ids = []
        for sub_token in root_sub_tokens:
            root_sub_token_ids.append(self.look_up_for_id_of_token(sub_token))
       
        root_json = {
            "node_type": str(root.srcml_kind),
            "node_token": root_sub_token_ids,
            "node_token_text": str(root.text),
            "children": []
        }

        queue_json = [root_json]
      
        while queue:
        
            current_node = queue.pop(0)
            num_nodes += 1
            current_node_json = queue_json.pop(0)

            children = [x for x in current_node.child]
            queue.extend(children)
        
            for child in children:
                child_sub_tokens = identifier_splitting.split_identifier_into_parts(str(child.text))
                child_sub_tokens = remove_noisy_tokens(child_sub_tokens)
                tree_tokens.extend(child_sub_tokens)

                children_sub_token_ids = []
                for sub_token in child_sub_tokens:
                    sub_token = process_token(sub_token)
                    children_sub_token_ids.append(self.look_up_for_id_of_token(sub_token))

                # To limit the number of sub tokens to 8 to reduce computation intensity
                children_sub_token_ids = list(set(children_sub_token_ids))
                # if len(children_sub_token_ids) > 8:
                # children_sub_token_ids = random.sample(children_sub_token_ids, 8)

                if len(children_sub_token_ids) == 0:
                    children_sub_token_ids.append(0)
               
                # print(children_sub_token_ids)
                child_json = {
                    "node_type": str(child.srcml_kind),
                    "node_token": children_sub_token_ids,
                    "node_token_text":str(child.text),
                    "children": []
                }

                current_node_json['children'].append(child_json)
                queue_json.append(child_json)

        tree_tokens = list(set(tree_tokens))
        
        # print(node_token_root_json)
        return root_json, num_nodes, tree_tokens
    
    # Not really put the trees into buckets, only put the tree_path and the sub_id into buckets to reduce the cost of computation
    def put_trees_into_bucket(self, trees):
        bucket_sizes = np.array(list(range(30 , 7500 , 10)))

        # Maintain two copy of the buckets of the same dataset for different purposes
        train_buckets = defaultdict(list)
        val_buckets = defaultdict(list)

        for tree_path, tree_data in trees.items():
            tree_size = tree_data["size"]
            chosen_bucket_idx = np.argmax(bucket_sizes > tree_size)
            print("Putting these ids to bucket : " + str(tree_data["subtrees_ids"]))
            for subtree_id in tree_data["subtrees_ids"]:
                temp_bucket_data = {}
                temp_bucket_data["file_path"] = tree_path
                temp_bucket_data["subtree_id"] = subtree_id
                
                train_buckets[chosen_bucket_idx].append(temp_bucket_data)

            val_buckets[chosen_bucket_idx].append(temp_bucket_data)

        return train_buckets, val_buckets, bucket_sizes
