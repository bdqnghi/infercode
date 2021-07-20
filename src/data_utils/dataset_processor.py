import numpy as np
import os
from tqdm import *
#import pickle
from .vocabulary import Vocabulary
from .ast_util import ASTUtil
from collections import defaultdict
import pickle
import logging

class DatasetProcessor():
    
    LOGGER = logging.getLogger('DatasetProcessor')
    def __init__(self, input_data_path: str, output_tensors_path: str, 
                node_type_vocab_model_path: str, node_token_vocab_model_path: str, 
                subtree_vocab_model_path: str, 
                ast_util: ASTUtil):
        
        self.input_data_path = input_data_path
        self.output_tensors_path = output_tensors_path
        self.node_type_vocab_model_path = node_type_vocab_model_path
        self.node_token_vocab_model_path = node_token_vocab_model_path
        self.subtree_vocab_model_path = subtree_vocab_model_path
        self.subtree_vocab = Vocabulary(10000, subtree_vocab_model_path)

        self.ast_util = ast_util
        # self.ast_util = ASTUtil(node_type_vocab_model_path=node_type_vocab_model_path, 
                                        # node_token_vocab_model_path=node_token_vocab_model_path, language=language)
        

    def process(self):

        bucket_sizes = np.array(list(range(20 , 7500 , 20)))
        buckets = defaultdict(list)

        for subdir , dirs, files in os.walk(self.input_data_path): 
            for file in tqdm(files):
                
                file_path = os.path.join(subdir, file)
                
                with open(file_path, "rb") as f:
                    code_snippet = f.read()

                tree_representation, tree_size = self.ast_util.simplify_ast(code_snippet)

                tree_indexes = self.ast_util.transform_tree_to_index(tree_representation)
                tree_indexes["size"] = tree_size 

                # Extract all subtrees from the code snippet
                subtrees = self.ast_util.extract_subtrees(code_snippet)
                
                # ----------Convert subtree strings to id----------
                subtrees_id = []
                for subtree in subtrees:
                    subtree_str = "-".join(subtree)
                    subtree_id = self.subtree_vocab.get_id_or_unk_for_text(subtree_str)
                    if len(subtree_id) == 1 and subtree_id[0] != 0:
                        subtrees_id.append(subtree_id[0])

                subtrees_id = list(set(subtrees_id))
                # --------------------------------------------------
                
                # Put different instances of the same snippet (with different subtree id) into buckets for training
                for subtree_id in subtrees_id:
                    tree_indexes["subtree_id"] = subtree_id
                    chosen_bucket_idx = np.argmax(bucket_sizes > tree_size)
                    buckets[chosen_bucket_idx].append(tree_indexes)

                # count = count + 1

        self.LOGGER.info("Saving processed data into pickle format.....")
        pickle.dump(buckets, open(self.output_tensors_path, "wb" ) )

        return buckets



  