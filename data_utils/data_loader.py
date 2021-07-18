from tqdm import *
import random
import logging
import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))

class DataLoader():
    LOGGER = logging.getLogger('DataLoader')

    def __init__(self, ast_util, batch_size):
        self.ast_util = ast_util
        self.batch_size = batch_size

    def make_minibatch_iterator(self, buckets):
        bucket_ids = list(buckets.keys())
        random.shuffle(bucket_ids)
        for bucket_idx in bucket_ids:

            trees = buckets[bucket_idx]
            random.shuffle(trees)
            
            batch_trees = []
            samples = 0
            for i, tree in enumerate(trees):
                batch_trees.append(tree)
                samples += 1
                if samples >= self.batch_size:
                    batch_obj = self.ast_util.trees_to_batch_tensors(batch_trees)
                
                    yield batch_obj
                    batch_trees = []
                    samples = 0

