import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from os import path
from .vocabulary import Vocabulary
from tree_sitter import Language, Parser
from pathlib import Path
import glob, os
import numpy as np


class TensorUtil():


    def __init__(self):
        pass
    
    def transform_tree_to_index(self, tree):
       
        # print(tree)
        node_type = []
        node_type_id = []
        node_tokens = []
        node_tokens_id = []
        node_index = []
        children_index = []
        children_node_type = []
        children_node_type_id = []
        children_node_tokens = []
        children_node_tokens_id = []

        queue = [(tree, -1)]
        # print queue
        while queue:
            # print "############"
            node, parent_ind = queue.pop(0)
            # print node
            # print parent_ind
            node_ind = len(node_type)
            # print "node ind : " + str(node_ind)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children_index.append([])
            children_node_type.append([])
            children_node_type_id.append([])
            children_node_tokens.append([])
            children_node_tokens_id.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children_index[parent_ind].append(node_ind)
                children_node_type[parent_ind].append(node["node_type"])
                children_node_type_id[parent_ind].append(int(node["node_type_id"]))
                children_node_tokens[parent_ind].append(node["node_tokens"])
                children_node_tokens_id[parent_ind].append(node["node_tokens_id"])
            
        
            node_type.append(node["node_type"])
            node_type_id.append(node["node_type_id"])
            node_tokens.append(node["node_tokens"])
            node_tokens_id.append(node["node_tokens_id"]) 
            node_index.append(node_ind)

        data = {}
        data["node_index"] = node_index
        data["node_type"] = node_type
        data["node_type_id"] = node_type_id
        data["node_tokens"] = node_tokens
        data["node_tokens_id"] = node_tokens_id
        data["children_index"] = children_index
        data["children_node_type"] = children_node_type
        data["children_node_type_id"] = children_node_type_id
        data["children_node_tokens"] = children_node_tokens
        data["children_node_tokens_id"] = children_node_tokens_id
        
        return data

    def trees_to_batch_tensors(self, all_tree_indices):
        batch_node_index = []
        batch_node_type = []
        batch_node_type_id = []
        batch_node_tokens = []
        batch_node_tokens_id = []
        batch_children_index = []
        batch_children_node_type = []
        batch_children_node_type_id = []
        batch_children_node_tokens = []
        batch_children_node_tokens_id = []
        batch_subtree_id = []
        batch_language_index = []

        for tree_indices in all_tree_indices:
            
            # random_subtree_id = self.random_sampling_subtree(tree_data["subtrees_ids"])

            batch_node_index.append(tree_indices["node_index"])
            batch_node_type.append(tree_indices["node_type"])
            batch_node_type_id.append(tree_indices["node_type_id"])
            batch_node_tokens.append(tree_indices["node_tokens"])
            batch_node_tokens_id.append(tree_indices["node_tokens_id"])
            batch_children_index.append(tree_indices["children_index"])
            batch_children_node_type.append(tree_indices["children_node_type"])
            batch_children_node_type_id.append(tree_indices["children_node_type_id"])
            batch_children_node_tokens.append(tree_indices["children_node_tokens"])
            batch_children_node_tokens_id.append(tree_indices["children_node_tokens_id"])

            if "subtree_id" in tree_indices:
                batch_subtree_id.append(tree_indices["subtree_id"])
            # batch_subtree_id.append([5, 2])
        
        # [[]]
        batch_node_index = self._pad_batch_2D(batch_node_index)
        # [[]]
        batch_node_type_id = self._pad_batch_2D(batch_node_type_id)
        # [[[]]]
        batch_node_tokens_id = self._pad_batch_3D(batch_node_tokens_id)
        # [[[]]]
        batch_children_index = self._pad_batch_3D(batch_children_index)
        # [[[]]]
        batch_children_node_type_id = self._pad_batch_3D(batch_children_node_type_id)    
        # [[[[]]]]
        batch_children_node_tokens_id = self._pad_batch_4D(batch_children_node_tokens_id)
        
        batch_obj = {
            "batch_node_index": np.asarray(batch_node_index),
            "batch_node_type_id": np.asarray(batch_node_type_id),
            "batch_node_tokens_id": np.asarray(batch_node_tokens_id),
            "batch_children_index": np.asarray(batch_children_index),
            "batch_children_node_type_id": np.asarray(batch_children_node_type_id),
            "batch_children_node_tokens_id": np.asarray(batch_children_node_tokens_id),
            "batch_language_index": np.asarray(batch_language_index)
        }

        # These item does not need to be converted to numpy array, they are for debugging purpose only
        batch_obj["batch_node_type"] = batch_node_type
        batch_obj["batch_node_tokens"] = batch_node_tokens
        batch_obj["batch_children_node_tokens"] = batch_children_node_tokens

        if len(batch_subtree_id) != 0:
            batch_subtree_id = np.reshape(batch_subtree_id, (len(all_tree_indices), 1))

        batch_obj["batch_subtree_id"] = batch_subtree_id

        return batch_obj

  
    def _pad_batch_2D(self, batch):
        max_batch = max([len(x) for x in batch])
        batch = [n + [0] * (max_batch - len(n)) for n in batch]
        batch = np.asarray(batch)
        return batch

    def _pad_batch_3D(self, batch):
        max_2nd_D = max([len(x) for x in batch])
        max_3rd_D = max([len(c) for n in batch for c in n])
        batch = [n + ([[]] * (max_2nd_D - len(n))) for n in batch]
        batch = [[c + [0] * (max_3rd_D - len(c)) for c in sample] for sample in batch]
        batch = np.asarray(batch)
        return batch


    def _pad_batch_4D(self, batch):
        max_2nd_D = max([len(x) for x in batch])
        max_3rd_D = max([len(c) for n in batch for c in n])
        max_4th_D = max([len(s) for n in batch for c in n for s in c])
        batch = [n + ([[]] * (max_2nd_D - len(n))) for n in batch]
        batch = [[c + ([[]] * (max_3rd_D - len(c))) for c in sample] for sample in batch]
        batch = [[[s + [0] * (max_4th_D - len(s)) for s in c] for c in sample] for sample in batch]
        batch = np.asarray(batch)
        return batch

