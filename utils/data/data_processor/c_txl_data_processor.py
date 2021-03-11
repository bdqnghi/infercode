"""
The interface to parse the TXL Representation of C (in form of XML)
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

class CTxlDataProcessor(DataProcessor):
   
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
                if file.endswith(".xml"):
                    try:
                        txl_file_path = os.path.join(subdir,file)
                        # print(txl_file_path)
                        txl_file_path_splits = txl_file_path.split("/")
                        label = int(txl_file_path_splits[len(txl_file_path_splits)-2]) - 1
                        # print(pkl_file_path)

                        with open(txl_file_path) as f:
                            line_count = sum(1 for _ in f)

                        if line_count < 10000:
                            count_processed_files += 1
                            xml_representation = ET.parse(txl_file_path)
                            tree, sub_tokens, size = self._convert_ast_into_simpler_tree_format(xml_representation.getroot())

                            # print(tree)
                            # print(tree)
                            tree_data = {
                                "tree": tree,
                                "size": size,
                                "label": label,
                                "sub_tokens": sub_tokens,
                                "file_path": txl_file_path
                            }
                            trees.append(tree_data)

                            # all_tokens.extend(sub_tokens)

                        else:
                            print("Ignoring too large file with path : " + txl_file_path)
                            
                    except Exception as e:
                        print(e)
                

        # all_tokens = list(set(all_tokens))
        print("Total processed files : " + str(count_processed_files))
        return trees

    def _convert_ast_into_simpler_tree_format(self, root):
        num_nodes = 0

        queue = [root]

        # Extract root token information, include splitting a token into sub tokens
        root_token = self.process_token(str(root.text))
        if root_token is None or root_token == "None": 
            root_token = ""
        root_sub_tokens = identifier_splitting.split_identifier_into_parts(root_token)
        root_sub_tokens = self.remove_noisy_tokens(root_sub_tokens)
        
        root_sub_token_ids = []
        for sub_token in root_sub_tokens:
            root_sub_token_ids.append(self.look_up_for_id_from_token(sub_token))

        root_json = {
            "node_type": str(root.tag),
            "node_type_id": self.look_up_for_id_from_node_type(str(root.tag)),
            "node_token": root_token,
            "node_sub_tokens": root_sub_tokens,
            "node_sub_tokens_id": root_sub_token_ids,
            "children": None # Using children = None instead of [] to avoid the error 'Python 3: maximum recursion depth exceeded'
        }

      
        tree_tokens = []
        tree_tokens.extend(root_sub_tokens)
        
        queue_json = [root_json]
        while queue:
        
            current_node = queue.pop(0)
            current_node_json = queue_json.pop(0)
            num_nodes += 1

            children = [x for x in current_node]
            queue.extend(children)

            if len(children) > 0:
                current_node_json['children'] = []

            for child in children:

                child_token = self.process_token(str(child.text))
                if child_token is None or child_token == "None": 
                    child_token = ""
                # else:

                child_sub_tokens = identifier_splitting.split_identifier_into_parts(str(child_token))
                child_sub_tokens = self.remove_noisy_tokens(child_sub_tokens)

                children_sub_token_ids = []
                for sub_token in child_sub_tokens:
                    sub_token = self.process_token(sub_token)
                    # print(sub_token)
                    sub_token_id = self.look_up_for_id_from_token(sub_token)
                    children_sub_token_ids.append(sub_token_id)

                if len(children_sub_token_ids) == 0:
                    children_sub_token_ids.append(0)

                child_json = {
                    "node_type": str(child.tag),
                    "node_type_id": self.look_up_for_id_from_node_type(str(child.tag)),
                    "node_token": child_token,
                    "node_sub_tokens": child_sub_tokens,
                    "node_sub_tokens_id": children_sub_token_ids,
                    "children": []
                }

                tree_tokens.extend(child_sub_tokens)

                current_node_json['children'].append(child_json)
                queue_json.append(child_json)

        tree_tokens = list(set(tree_tokens))

        return root_json, tree_tokens, num_nodes

  
            

   