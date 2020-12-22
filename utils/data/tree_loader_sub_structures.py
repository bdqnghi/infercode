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
import pyarrow
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle
from utils import identifier_splitting
from tensorflow.keras.preprocessing.sequence import pad_sequences
import hashlib

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


class CodeClassificationData():
   
    def __init__(self, opt, data_path, is_training=True, is_testing=False, is_validating=False):
        if is_training:
            print("Processing training data....")
           
        if is_testing:
            print("Processing testing data....")
     
        if is_validating:
            print("Processing validation data....")
         
        self.is_training = is_training
        self.is_testing = is_testing
        self.is_validating = is_validating
        self.batch_size = opt.batch_size
    
        self.node_type_lookup = opt.node_type_lookup
        self.node_token_lookup = opt.node_token_lookup
        self.subtree_lookup = opt.subtree_lookup

        self.num_subtrees = len(self.subtree_lookup.keys())

        self.tree_size_threshold_upper = opt.tree_size_threshold_upper
        self.tree_size_threshold_lower = opt.tree_size_threshold_lower
        self.num_files_threshold = opt.num_files_threshold

        self.num_sampling = opt.num_sampling

        base_name =os.path.basename(data_path)
        parent_base_name = os.path.basename(os.path.dirname(data_path))
        base_path = str(os.path.dirname(data_path))
        saved_input_filename = "%s/%s-%s.pkl" % (base_path, parent_base_name, base_name)

        if os.path.exists(saved_input_filename):
            print("Loading existing data file: ", str(saved_input_filename))
            self.data = pickle.load(open(saved_input_filename, "rb"))
           

        else:
            trees = self.load_program_data(data_path)
            self.trees = trees
            self.data = self.process_raw_trees()
            print("Serializing......")
            pickle.dump(self.data, open(saved_input_filename, "wb" ) )


    def load_program_data(self, directory):
        trees = []
        count = 0
        for subdir , dirs, files in os.walk(directory): 
            for file in tqdm(files):
                
                if file.endswith(".pkl") and not file.endswith(".slice.pkl"):
                    pkl_file_path = os.path.join(subdir,file)
                    print(pkl_file_path)
                    pkl_file_path_splits = pkl_file_path.split("/")
                    label = int(pkl_file_path_splits[len(pkl_file_path_splits)-2]) - 1
                    # print(pkl_file_path)
                    pb_representation = self.load_tree_from_pickle_file(pkl_file_path)
                    # print(pb_representation)
                    root = pb_representation.element
      
                    tree, size, tokens = self._traverse_tree(root)
                    subtrees = self.get_subtrees(root)

                    # print(tokens)
                    tree_data = {
                        "tree": tree,
                        "tokens": tokens,
                        "subtrees": subtrees,
                        "size": size,
                        "label": label,
                        "file_path": pkl_file_path
                    }
                    trees.append(tree_data)
                          
        return trees

    def look_up_for_id_of_token(self, token):
        
        token_id = self.node_token_lookup["<SPECIAL>"]
        if token in self.node_token_lookup:
            token_id = self.node_token_lookup[token]

        return token_id

    def look_up_for_token_of_id(self, token_id):
        token = self.node_token_lookup.inverse[token_id]
        return token

    def load_tree_from_pickle_file(self, file_path):
        """Builds an AST from a script."""
   
        with open(file_path, 'rb') as file_handler:
            tree = pickle.load(file_handler)
            # print(tree)
            return tree
        return "error"

    def get_children_nodes(self, node):

        children_nodes = []
        for child in node.child:

            child_sub_tokens = identifier_splitting.split_identifier_into_parts(str(child.text))
            child_sub_tokens = remove_noisy_tokens(child_sub_tokens)

            children_sub_token_ids = []
            for sub_token in child_sub_tokens:
                sub_token = process_token(sub_token)
                children_sub_token_ids.append(self.look_up_for_id_of_token(sub_token))

            children_sub_token_ids = list(set(children_sub_token_ids))
            children_sub_token_ids.sort()

            children_node = {
                "node_type": child.kind,
                "node_token": children_sub_token_ids,
                "node_token_text": str(child.text),
            }

            children_nodes.append(children_node)

        return children_nodes

    def flatten_features(self, subtree):
        parent_features = [subtree["node_type"]]
        children_features = []

        for child in subtree["children"]:
            children_features.append(child["node_type"])

        children_features.sort()

        features = parent_features + children_features

        return features

    def get_subtrees(self, root):

        subtrees = []

        queue = [root]


        while queue:
            current_node = queue.pop(0)
            queue.extend(current_node.child)
            
            children_nodes = self.get_children_nodes(current_node)


            current_node_token = str(current_node.text)
            current_node_sub_tokens = identifier_splitting.split_identifier_into_parts(current_node_token)
            current_node_sub_tokens = remove_noisy_tokens(current_node_sub_tokens)

            current_node_sub_token_ids = []
            for sub_token in current_node_sub_tokens:
                current_node_sub_token_ids.append(self.look_up_for_id_of_token(sub_token))

            current_node_sub_token_ids = list(set(current_node_sub_token_ids))
            current_node_sub_token_ids.sort()

            subtree = {
                    "node_type": current_node.kind,
                    "node_token": current_node_sub_token_ids,
                    "node_token_text": str(current_node.text),
                    'children': children_nodes
                    #"children": [x.kind for x in child.child]
            }

            # print(sub)
            features = self.flatten_features(subtree)

            if len(features) > 3:
                str_features = "_".join(map(str,features))
                hash_object = hashlib.md5(str_features.encode()).hexdigest()

                if hash_object in self.subtree_lookup:
                    print("Hash object : " + str(hash_object) + " is existing")
                    subtrees.append(self.subtree_lookup[hash_object])
                    # subtrees.append(hash_object)
                else:
                    print("Hash object : " + str(hash_object) + " is not existing")

        subtrees = list(set(subtrees))
        print(subtrees)

        return subtrees

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
            "node_type": str(root.kind),
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
                    "node_type": str(child.kind),
                    "node_token": children_sub_token_ids,
                    "node_token_text":str(child.text),
                    "children": []
                }

                current_node_json['children'].append(child_json)
                queue_json.append(child_json)

        tree_tokens = list(set(tree_tokens))
        
        # print(node_token_root_json)
        return root_json, num_nodes, tree_tokens
    
    def process_raw_trees(self):
        bucket_sizes = np.array(list(range(30 , 7500 , 10)))
        buckets = defaultdict(list)

        for tree_data in self.trees:
          
            tree, tree_size = tree_data["tree"], tree_data["size"]

            chosen_bucket_idx = np.argmax(bucket_sizes > tree_size)
            buckets[chosen_bucket_idx].append(tree_data)
     
        return buckets, bucket_sizes


    def extract_training_data(self, tree_data):
        
        tree, label, tokens, size, file_path = tree_data["tree"], tree_data["label"], tree_data["tokens"] , tree_data["size"], tree_data["file_path"]
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

        # print("-------------")
        # print(node_types)
        # print(node_tokens)
        # print(children_indices)
        # print(children_node_types)
        # print(children_node_tokens)
        token_ids = []
        token_ids.append(self.look_up_for_id_of_token("<GO>"))
        for token in tokens:
            token_ids.append(self.look_up_for_id_of_token(token))
        token_ids.append(self.look_up_for_id_of_token("<EOS>"))
     
        return node_indexes, node_types, node_tokens, node_tokens_text, children_indices, children_node_types, children_node_tokens, token_ids, label, size, file_path

    def random_sampling_subtree(self,subtrees):
        return random.choice(subtrees)

    def random_sampling_multiple_subtrees(self,subtrees, num_sampling):
        return random.sample(subtrees, num_sampling)
                
    def make_batch(self, batch_data):
        batch_node_indexes = []
        batch_node_types = []
        batch_node_tokens = []
        batch_node_tokens_text = []
        batch_children_indices = []
        batch_children_node_types = []
        batch_children_node_tokens = []
        batch_labels = []
        batch_tree_size = []
        batch_file_path = []
        batch_token_ids = []
        batch_length_targets = []
        batch_subtrees_ids = []
        batch_sampled_subtrees_ids = []
        batch_subtrees_ids_onehot = []
        batch_subtrees_ids_soft_onehot = []
        for tree_data in batch_data:
            node_indexes, node_types, node_tokens, node_tokens_text, children_indices, children_node_types, children_node_tokens, token_ids, label, size, file_path = self.extract_training_data(tree_data)
            
            subtree_id = self.random_sampling_subtree(tree_data["subtrees"])
            sampled_subtrees_ids = self.random_sampling_multiple_subtrees(tree_data["subtrees"], self.num_sampling)

            batch_subtrees_ids.append(subtree_id)
            batch_sampled_subtrees_ids.append(sampled_subtrees_ids)
            batch_subtrees_ids_onehot.append(_onehot(subtree_id, self.num_subtrees))
            batch_subtrees_ids_soft_onehot.append(_soft_onehot(sampled_subtrees_ids, self.num_subtrees))

            batch_node_indexes.append(node_indexes)
            batch_node_types.append(node_types)
            batch_node_tokens.append(node_tokens)
            batch_node_tokens_text.append(node_tokens_text)
            batch_children_indices.append(children_indices)
            batch_children_node_types.append(children_node_types)
            batch_children_node_tokens.append(children_node_tokens)
            batch_labels.append(label)
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
            "batch_labels": batch_labels,
            "batch_tree_size": batch_tree_size,
            "batch_file_path": batch_file_path,
            "batch_subtrees_ids": batch_subtrees_ids,
            "batch_sampled_subtrees_ids": batch_sampled_subtrees_ids,
            "batch_subtrees_ids_onehot": batch_subtrees_ids_onehot,
            "batch_subtrees_ids_soft_onehot": batch_subtrees_ids_soft_onehot, 
        }
        return batch_obj

    #  def _pad_batch(self, nodes, children, labels):
    #     if not nodes:
    #         return [], [], []
    #     max_nodes = max([len(x) for x in nodes])
    #     max_children = max([len(x) for x in children])
    #     child_len = max([len(c) for n in children for c in n])

    #     nodes = [n + [0] * (max_nodes - len(n)) for n in nodes]
    #     # pad batches so that every batch has the same number of nodes
    #     children = [n + ([[]] * (max_children - len(n))) for n in children]
    #     # pad every child sample so every node has the same number of children
    #     children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]


    #     return nodes, children, labels

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

    def _onehot(self, i, total):
        return [1.0 if j == i else 0.0 for j in range(total)]

    def _produce_mask_vector(self, nodes):
        masks = []

        for n in nodes:        
            mask = [1 for i in range(len(n))]
            masks.append(mask)

        padded_inputs = pad_sequences(masks, padding='post')
        return padded_inputs

    def make_minibatch_iterator(self):
        
        (buckets, bucket_sizes) = self.data
        bucket_ids = list(buckets.keys())
        random.shuffle(bucket_ids)
        # for bucket_idx, bucket_data in buckets.items():
        for bucket_idx in bucket_ids:
            bucket_data = buckets[bucket_idx]
            # print(file)
            # print("Shuffling data.....")
            random.shuffle(bucket_data)
            
            elements = []
            samples = 0
    
            for i, tree_data in enumerate(bucket_data):
            
                _, label, size = tree_data["tree"], tree_data["label"], tree_data["size"]
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
                    batch["batch_labels"] = np.asarray(batch_obj["batch_labels"])
                    batch["batch_tree_size"] = batch_obj["batch_tree_size"]
                    batch["batch_file_path"] = batch_obj["batch_file_path"]
                    batch["batch_node_indicators"] = batch_node_indicators
                    batch["batch_token_ids"] = batch_obj["batch_token_ids"]
                    batch["batch_length_targets"] = batch_obj["batch_length_targets"]
                    batch["batch_subtrees_ids"] = np.reshape(batch_obj["batch_subtrees_ids"], (self.batch_size, 1))
                    batch["batch_sampled_subtrees_ids"] = np.reshape(batch_obj["batch_sampled_subtrees_ids"], (self.batch_size, self.num_sampling))
                    batch["batch_subtrees_ids_onehot"] = np.asarray(batch_obj["batch_subtrees_ids_onehot"]),
                    batch["batch_subtrees_ids_soft_onehot"] = np.asarray(batch_obj["batch_subtrees_ids_soft_onehot"])

            
                    yield batch
                    elements = []
                    samples = 0
