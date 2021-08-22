from tqdm import *
import random
import logging
import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from .tensor_util import TensorUtil

class DataLoader():
    LOGGER = logging.getLogger('DataLoader')

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.tensor_util = TensorUtil()

    def make_minibatch_iterator(self, buckets):
        bucket_ids = list(buckets.keys())
        random.shuffle(bucket_ids)
        for bucket_idx in bucket_ids:
            self.LOGGER.debug("Switching bucket...")
            trees = buckets[bucket_idx]
            self.LOGGER.debug(f"Num items in bucket {len(trees)}")
            random.shuffle(trees)
            
            batch_trees = []
            samples = 0
            for i, tree in enumerate(trees):
                if tree["size"] < 500:
                    batch_trees.append(tree)
                    samples += 1
                
                if samples >= self.batch_size:
                    batch_obj = self.tensor_util.trees_to_batch_tensors(batch_trees)
                
                    yield batch_obj
                    batch_trees = []
                    samples = 0

