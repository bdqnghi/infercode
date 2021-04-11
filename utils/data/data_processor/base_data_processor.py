"""
Base inferface to parse AST Representation of code 
"""
import os
import re
import numpy as np
from collections import defaultdict
from bidict import bidict
import pickle

excluded_tokens = [",","{",";","}",")","(",'"',"'","`",""," ","[]","[","]","/",":",".","''","'.'", "\\", "'['", "']","''","_","__"]

class DataProcessor():
   
    def __init__(self, node_type_vocab_path, token_vocab_path, subtree_vocab_path, data_path):
        
        self.node_type_vocab_path = node_type_vocab_path
        self.token_vocab_path = token_vocab_path
        self.data_path = data_path

        self.node_token_lookup = self.load_node_token_vocab(token_vocab_path)
        self.node_type_lookup = self.load_node_type_vocab(node_type_vocab_path)
        self.subtree_lookup = self.load_subtree_vocab(subtree_vocab_path)

        print(self.node_type_lookup)
        base_name =os.path.basename(data_path)
        
        self.simple_tree_pkl_name = os.path.basename(os.path.dirname(data_path))
        self.buckets_name = os.path.basename(os.path.dirname(data_path))

        base_path = str(os.path.dirname(data_path))
        self.simple_tree_pkl_path = "%s/%s-%s.pkl" % (base_path, "trees", base_name)
        self.all_subtrees_buckets_name_path = "%s/%s-%s.pkl" % (base_path, "buckets-all", base_name)
        # self.random_subtrees_buckets_name_path = "%s/%s-%s.pkl" % (base_path, "buckets-random", base_name)

        self.bucket_sizes = np.array(list(range(30 , 7500 , 10)))
        self.buckets = defaultdict(list)


        if os.path.exists(self.simple_tree_pkl_path):
            print("Simple trees path exists, loading the pickle....")
            self.trees = pickle.load(open(self.simple_tree_pkl_path, "rb" ))
        else:
            print("Simple trees path does not exist, loading from raw files....")
            self.trees = self.load_program_data(self.data_path)
            pickle.dump(self.trees, open(self.simple_tree_pkl_path, "wb" ) )

        print("Convert trees into training indices....")
        self.all_subtrees_buckets, self.bucket_sizes = self.convert_trees_into_training_indices(self.trees)

        pickle.dump(self.all_subtrees_buckets, open(self.all_subtrees_buckets_name_path, "wb" ) )
        # pickle.dump(self.random_subtrees_buckets, open(self.random_subtrees_buckets_name_path, "wb" ) )


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
                node_type_lookup[line] = i

        return bidict(node_type_lookup)

    def load_subtree_vocab(self, subtree_vocab_path):
        subtree_lookup = {}
        with open(subtree_vocab_path, "r") as f:
            data = f.readlines()
           
            for i, line in enumerate(data):
                line = line.replace("\n", "").strip()
                subtree_lookup[line] = i

        return bidict(subtree_lookup)

    def process_token(self, token):
        for t in excluded_tokens:
            token = token.replace(t, "")
            # token = re.sub(r'[^\w]', ' ', token)
        return token

    def remove_noisy_tokens(self, tokens):
        temp_tokens = []
        for t in tokens:
            t = self.process_token(t)
            if t:
                temp_tokens.append(t)
        return temp_tokens

    def look_up_for_id_from_token(self, token):
        token_id = self.node_token_lookup["<SPECIAL>"]
        if token in self.node_token_lookup:
            token_id = self.node_token_lookup[token]

        return token_id

    def look_up_for_token_from_id(self, token_id):
        return self.node_token_lookup.inverse[token_id]

    def look_up_for_id_from_node_type(self, node_type):
        # node_type = node_type.upper()
        node_type_id = self.node_type_lookup[node_type]
        return node_type_id

    def look_up_for_node_type_from_id(self, node_type_id):
        return self.node_type_lookup.inverse[node_type_id]

    def look_up_for_id_from_subtree(self, subtree):
        # node_type = node_type.upper()
        subtree_id = self.subtree_lookup[subtree]
        return subtree_id

    def look_up_for_subtree_from_id(self, subtree_id):
        return self.subtree_lookup.inverse[subtree_id]

    def save_tokens_vocab(self, tokens, token_vocab_path):
        tokens.sort()
        with open(token_vocab_path, "w") as f:
            f.write("<SPECIAL>")
            f.write("\n")
            for t in tokens:
                f.write(t)
                f.write("\n")

    def process_list_of_sub_tokens(self, sub_tokens):
        sub_tokens = list(filter(None, sub_tokens))
        sub_tokens = list(map(lambda x: x.lower(), sub_tokens))
        temp_sub_tokens = []
        for s in sub_tokens:
            if not self.detect_special_characer(s):
                temp_sub_tokens.append(s)
        return temp_sub_tokens

    def detect_special_characer(self, pass_string): 
        regex= re.compile('"[@_!#$%^&*()<>?/\|}{~:]') 
        if re.match("^[a-zA-Z0-9_]*$", pass_string):
            res = False
        else: 
            res = True
        return res

    def load_tree_from_pickle_file(self, file_path):
        """Builds an AST from a script."""
   
        with open(file_path, 'rb') as file_handler:
            tree = pickle.load(file_handler)
            # print(tree)
            return tree
        return None


    # Prepare tensor data for training
    # Not really put the trees into buckets, only put the tree_path and the sub_id into buckets to reduce the cost of computation
    def convert_trees_into_training_indices(self, trees):

        bucket_sizes = np.array(list(range(30 , 7500 , 10)))
        # Maintain two copy of the buckets of the same dataset for different purposes
        all_subtrees_buckets = defaultdict(list)
        # random_subtrees_buckets = defaultdict(list)

        for tree_path, tree_data in trees.items():
            tree_size = tree_data["size"]
            chosen_bucket_idx = np.argmax(bucket_sizes > tree_size)
            print("Putting these ids to bucket : " + str(tree_data["subtree_ids"]))
            for subtree_id in tree_data["subtree_ids"]:
                temp_bucket_data = {}
                temp_bucket_data["file_path"] = tree_path
                temp_bucket_data["subtree_id"] = subtree_id
                temp_bucket_data["size"] = tree_data["size"]

                all_subtrees_buckets[chosen_bucket_idx].append(temp_bucket_data)

            # random_subtrees_buckets[chosen_bucket_idx].append(temp_bucket_data)

        print(all_subtrees_buckets)
        # return all_subtrees_buckets, random_subtrees_buckets, bucket_sizes
        return all_subtrees_buckets, bucket_sizes


    def extract_training_data(self, tree_data):
        
        tree, subtree_ids, sub_tokens, size, file_path = tree_data["tree"], tree_data["subtree_ids"], tree_data["sub_tokens"] , tree_data["size"], tree_data["file_path"]
        # tree, sub_tokens, size, file_path = tree_data["tree"], tree_data["sub_tokens"] , tree_data["size"], tree_data["file_path"]
        print("Extracting............", file_path)
        # print(tree)
        node_type_id = []
        node_token = []
        node_sub_tokens_id = []
        node_index = []

        children_index = []
        children_node_type_id = []
        children_node_token = []
        children_node_sub_tokens_id = []
        # label = 0

        # print("Label : " + str(label))
        queue = [(tree, -1)]
        # print queue
        while queue:
            # print "############"
            node, parent_ind = queue.pop(0)
            # print node
            # print parent_ind
            node_ind = len(node_type_id)
            # print "node ind : " + str(node_ind)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children_index.append([])
            children_node_type_id.append([])
            children_node_token.append([])
            children_node_sub_tokens_id.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children_index[parent_ind].append(node_ind)
                children_node_type_id[parent_ind].append(int(node["node_type_id"]))
                children_node_token[parent_ind].append(node["node_token"])
                children_node_sub_tokens_id[parent_ind].append(node["node_sub_tokens_id"])
            # print("a")
            # print(children_node_types)
            # print("b")
            # print(children_node_sub_tokens_id)
            node_type_id.append(node['node_type_id'])
            node_token.append(node['node_token'])
            node_sub_tokens_id.append(node['node_sub_tokens_id'])
            node_index.append(node_ind)

        results = {}
        results["node_index"] = node_index
        results["node_type_id"] = node_type_id
        results["node_token"] = node_token
        results["node_sub_tokens_id"] = node_sub_tokens_id
        results["children_index"] = children_index
        results["children_node_type_id"] = children_node_type_id
        results["children_node_token"] = children_node_token
        results["children_node_sub_tokens_id"] = children_node_sub_tokens_id
        results["size"] = size
        results["subtree_ids"] = subtree_ids

        return results