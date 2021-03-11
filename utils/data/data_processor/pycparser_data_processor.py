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
from pycparser import parse_file
from collections import defaultdict
from util import identifier_splitting
import resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

class PycParserDataProcessor(DataProcessor):
   
    def __init__(self, node_type_vocab_path, token_vocab_path, data_path, parser):
        super().__init__(node_type_vocab_path, token_vocab_path, data_path, parser)

    def load_program_data(self, directory):
        trees = []
        
        count_processed_files = 0
        for subdir , dirs, files in os.walk(directory): 
            count = 0
            for file in tqdm(files):
                # if count < 20:
                if file.endswith(".c"):
                    try:
                        file_path = os.path.join(subdir,file)
                        # print(txl_file_path)
                        file_path_splits = file_path.split("/")
                        label = int(file_path_splits[len(file_path_splits)-2]) - 1
                        # print(pkl_file_path)

                        with open(file_path) as f:
                            line_count = sum(1 for _ in f)

                        # Set this condition to get rid of very large xml files (due to bugs when parsing)
                        if line_count < 10000:
                            count_processed_files += 1

                            ast = parse_file(file_path, use_cpp=True)
                            print(file_path)
                            tree, sub_tokens, size = self._convert_ast_into_simpler_tree_format(ast)

                            tree_data = {
                                "tree": tree,
                                "size": size,
                                "label": label,
                                "sub_tokens": sub_tokens,
                                "file_path": file_path
                            }
                            trees.append(tree_data)

                        else:
                            print("Ignoring too large file with path : " + txl_file_path)
                            
                    except Exception as e:
                        print(e)
                

        print("Total processed files : " + str(count_processed_files))
        return trees

    def _convert_ast_into_simpler_tree_format(self, root):
        """" Recursively convert an ast into dict representation. """
        num_nodes = 0
        klass = root.__class__
        queue = [root]

        root_token = ""
        for attr in klass.attr_names:
            attribute = getattr(root, attr)
            if attr == "name" or attr == "op" or attr == "declname":
                root_token = attribute
            if attr == "names":
                root_token = attribute[0]
        
        root_token = self.process_token(root_token)
        root_sub_tokens = identifier_splitting.split_identifier_into_parts(root_token)
        root_sub_tokens = self.remove_noisy_tokens(root_sub_tokens)


        root_sub_token_ids = []
        for sub_token in root_sub_tokens:
            root_sub_token_ids.append(self.look_up_for_id_from_token(sub_token))


        root_json = {
            "node_type": str(klass.__name__),
            "node_type_id": self.look_up_for_id_from_node_type(str(klass.__name__)),
            "node_token": root_token,
            "node_sub_tokens": root_sub_tokens,
            "node_sub_tokens_id": root_sub_token_ids,
            "children": []
        }


        tree_tokens = []
        tree_tokens.extend(root_sub_tokens)

        queue_json = [root_json]
        while queue:
            current_node = queue.pop(0)
            current_node_json = queue_json.pop(0)
            num_nodes += 1

            # Child attributes
            for child_name, child in current_node.children():
                
                child_token = ""
                for attr in child.__class__.attr_names:
                    attribute = getattr(child, attr)
                    if attr == "name" or attr == "op" or attr == "declname":
                        child_token = attribute
                    if attr == "names":
                        child_token = attribute[0]

                child_sub_tokens = identifier_splitting.split_identifier_into_parts(str(child_token))
            
                children_sub_token_ids = []
                for sub_token in child_sub_tokens:
                    sub_token = self.process_token(sub_token)
                    # print(sub_token)
                    sub_token_id = self.look_up_for_id_from_token(sub_token)
                    children_sub_token_ids.append(sub_token_id)

                if len(children_sub_token_ids) == 0:
                    children_sub_token_ids.append(0)

                child_json = {
                    "node_type": str(child.__class__.__name__),
                    "node_type_id": self.look_up_for_id_from_node_type(str(child.__class__.__name__)),
                    "node_token": child_token,
                    "node_sub_tokens": child_sub_tokens,
                    "node_sub_tokens_id": children_sub_token_ids,
                    "children": []
                }


                current_node_json['children'].append(child_json)
                queue_json.append(child_json)
                queue.append(child)
        

        tree_tokens = list(set(tree_tokens))

        # print(root_json)
        return root_json, tree_tokens, num_nodes

 