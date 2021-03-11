"""
The interface to parse the TXL Representation of Rust (in form of XML)
"""
import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
import os
import numpy as np
from .base_data_processor import DataProcessor
import xml.etree.ElementTree as ET
from tqdm import trange
from tqdm import *
import pickle
from util import identifier_splitting
import resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

class SrcmlDataProcessor(DataProcessor):
   
    def __init__(self, node_type_vocab_path, token_vocab_path, data_path, parser):
        super().__init__(node_type_vocab_path, token_vocab_path, data_path, parser)

    def load_program_data(self, directory):
        trees = []

        # all_tokens = []
        count_processed_files = 0
        for subdir , dirs, files in os.walk(directory): 
            count = 0
            for file in tqdm(files):
                # if count < 20:
                if file.endswith(".pickle"):
                    try:
                        pkl_file_path = os.path.join(subdir,file)
                        pkl_file_path_splits = pkl_file_path.split("/")
                        label = int(pkl_file_path_splits[len(pkl_file_path_splits)-2]) - 1

                        count_processed_files += 1

                        pb_representation = self.load_tree_from_pickle_file(pkl_file_path)
                        tree, sub_tokens, size = self._convert_ast_into_simpler_tree_format(pb_representation.element)
                            
                        tree_data = {
                            "tree": tree,
                            "size": size,
                            "label": label,
                            "sub_tokens": pkl_file_path,
                            "file_path": pkl_file_path
                        }
                        trees.append(tree_data)
      
                    except Exception as e:
                        print(e)
                
        # all_tokens = list(set(all_tokens))
        print("Total processed files : " + str(count_processed_files))
        return trees

    def _convert_ast_into_simpler_tree_format(self, root):
        num_nodes = 0

        queue = [root]

        root_token = str(root.text)
        root_sub_tokens = identifier_splitting.split_identifier_into_parts(root_token)
        root_sub_tokens = self.remove_noisy_tokens(root_sub_tokens)

        tree_tokens = []
        tree_tokens.extend(root_sub_tokens)

        root_sub_token_ids = []
        for sub_token in root_sub_tokens:
            root_sub_token_ids.append(self.look_up_for_id_from_token(sub_token))
       
        root_json = {
            "node_type": str(root.srcml_kind),
            "node_type_id": str(root.srcml_kind),
            "node_token": root_token,
            "node_sub_tokens": root_sub_tokens,
            "node_sub_tokens_id": root_sub_token_ids,
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
                child_sub_tokens = self.remove_noisy_tokens(child_sub_tokens)
                tree_tokens.extend(child_sub_tokens)

                children_sub_token_ids = []
                for sub_token in child_sub_tokens:
                    sub_token = self.process_token(sub_token)
                    children_sub_token_ids.append(self.look_up_for_id_from_token(sub_token))

                # To limit the number of sub tokens to 8 to reduce computation intensity
                children_sub_token_ids = list(set(children_sub_token_ids))
                # if len(children_sub_token_ids) > 8:
                # children_sub_token_ids = random.sample(children_sub_token_ids, 8)

                if len(children_sub_token_ids) == 0:
                    children_sub_token_ids.append(0)
               
                # print(children_sub_token_ids)
                child_json = {
                    "node_type": str(child.srcml_kind),  
                    "node_type_id": str(child.srcml_kind),
                    "node_token": str(child.text),
                    "node_sub_tokens": child_sub_tokens,
                    "node_sub_tokens_id": children_sub_token_ids,
                    "children": []
                }

                current_node_json['children'].append(child_json)
                queue_json.append(child_json)

        tree_tokens = list(set(tree_tokens))
        
        # print(node_token_root_json)
        return root_json, tree_tokens, num_nodes

  
            

   