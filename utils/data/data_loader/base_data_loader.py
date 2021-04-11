import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

class BaseDataLoader():
   
    def __init__(self, batch_size, tree_size_threshold_upper, tree_size_threshold_lower, tree_path, bucket_path, is_training=True):

        self.is_training = is_training
        self.buckets = pickle.load(open(bucket_path, "rb" ))
        self.trees = pickle.load(open(tree_path, "rb"))
        self.batch_size = batch_size
        self.tree_size_threshold_upper = tree_size_threshold_upper
        self.tree_size_threshold_lower = tree_size_threshold_lower
        # self.make_minibatch_iterator()
    
    def _onehot(self, i, total):
        zeros = np.zeros(total)
        zeros[i] = 1.0
        return zeros

    def make_batch(self, batch_data):
        batch_node_index = []
        batch_node_type_id = []
        batch_node_sub_tokens_id = []
        batch_node_token = []

        batch_children_index = []
        batch_children_node_type_id = []
        batch_children_node_sub_tokens_id = []
        batch_children_node_token = []

        batch_subtree_id = []
        batch_size = []

        for element in batch_data:
            tree_data = self.trees[element["file_path"]]
            
            batch_node_index.append(tree_data["node_index"])
            batch_node_type_id.append(tree_data["node_type_id"])
            batch_node_sub_tokens_id.append(tree_data["node_sub_tokens_id"])
            batch_node_token.append(tree_data["node_token"])

            batch_children_index.append(tree_data["children_index"])
            batch_children_node_type_id.append(tree_data["children_node_type_id"])
            batch_children_node_sub_tokens_id.append(tree_data["children_node_sub_tokens_id"])
            batch_children_node_token.append(tree_data["children_node_token"])

            batch_subtree_id.append(element["subtree_id"])
            batch_size.append(tree_data["size"])
        
        # [[]]
        batch_node_index = self._pad_batch_2D(batch_node_index)
        # [[]]
        batch_node_type_id = self._pad_batch_2D(batch_node_type_id)
        # [[[]]]
        batch_node_sub_tokens_id = self._pad_batch_3D(batch_node_sub_tokens_id)
        # [[[]]]
        batch_children_index = self._pad_batch_3D(batch_children_index)
        # [[[]]]
        batch_children_node_type_id = self._pad_batch_3D(batch_children_node_type_id)    
        # [[[[]]]]
        batch_children_node_sub_tokens_id = self._pad_batch_4D(batch_children_node_sub_tokens_id)

        batch_obj = {
            "batch_node_index": batch_node_index,
            "batch_node_type_id": batch_node_type_id,
            "batch_node_sub_tokens_id": batch_node_sub_tokens_id,
            "batch_children_index": batch_children_index,
            "batch_children_node_type_id": batch_children_node_type_id,
            "batch_children_node_sub_tokens_id": batch_children_node_sub_tokens_id,
            "batch_subtree_id": batch_subtree_id,
            "batch_size": batch_size
        }
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

    def make_minibatch_iterator(self):

        bucket_ids = list(self.buckets.keys())
        random.shuffle(bucket_ids)
        
        for bucket_idx in bucket_ids:

            bucket_data = self.buckets[bucket_idx]
            random.shuffle(bucket_data)
            
            elements = []
            samples = 0
      
            for i, ele in enumerate(bucket_data):
                if self.is_training == True:
                    if ele["size"] > self.tree_size_threshold_lower and ele["size"] < self.tree_size_threshold_upper:
                        elements.append(ele)
                        samples += 1
                else:
                    elements.append(ele)
                    samples += 1

              
                if samples >= self.batch_size:
                    batch_obj = self.make_batch(elements)                
                    yield batch_obj
                    elements = []
                    samples = 0



