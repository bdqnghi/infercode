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
import glob, os
from collections import defaultdict
from dpu_utils.codeutils import split_identifier_into_parts
from tree_sitter import Language, Parser
from utils.subtree_util import print_subtree
from os import path
import resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

class TreeSitterDataProcessor(DataProcessor):
   
    def __init__(self, node_type_vocab_path, token_vocab_path, data_path, parser):


        home = str(Path.home())
        cd = os.getcwd()
        os.chdir(path.join(home, ".tree-sitter", "bin"))
        self.Languages = {}
        for file in glob.glob("*.so"):
            try:
                lang = os.path.splitext(file)[0]
                self.Languages[lang] = Language(path.join(home, ".tree-sitter", "bin", file), lang)
            except:
                print("An exception occurred to {}".format(lang))
        os.chdir(cd)

        self.excluded_node_types = ["comment", "error", "'", '"']
        super().__init__(node_type_vocab_path, token_vocab_path, data_path, parser)

    
     
    def load_program_data(self, directory):
        trees = []
        
        # all_tokens = []
        count_processed_files = 0
        for subdir , dirs, files in os.walk(directory): 
            count = 0
            for file in tqdm(files):
                # if count < 20:
                file_path = os.path.join(subdir,file)
                print(file_path)
                # try:
                    # print(txl_file_path)
                file_path_splits = file_path.split("/")
        
                count_processed_files += 1

                parser = Parser()
                parser_lang = self.Languages.get("java")
                parser.set_language(parser_lang)

                binary_data = open(file_path, "rb").read()
                treesitter_tree = parser.parse(binary_data)

                tree, sub_tokens, size = self._convert_ast_into_simpler_tree_format(treesitter_tree.root_node, binary_data)

                subtrees_flattened = {}
                print_subtree(binary_data, treesitter_tree.root_node, subtrees_flattened, self.excluded_node_types)

                subtrees = []

                for key, subtree in subtrees_flattened.items():
                    print("--------------------")
                    nodes = subtree.split(",")
                    nodes = nodes[:len(nodes)-1]
                    print(nodes)
                    subtree_arr = []
                    for node in nodes:
                        node_info = node.split("-")
                        if len(node_info) >= 2 and len(node) > 1:
                            node_type = node_info[1]
                            if node_type:
                                # print("type:", node_type)
                                subtree_arr.append(node_type)
                    subtree_str = "_".join(subtree_arr)
                    subtrees.append(subtree_str)


                print(sub_tokens)
                tree_data = {
                    "tree": tree,
                    "size": size,
                    "subtrees": subtrees,
                    "sub_tokens": sub_tokens,
                    "file_path": file_path
                }
                trees.append(tree_data)
            
                        
                # except Exception as e:
                #     print("Exeception when processing the file :", file_path, "with exeption", str(e))
            

        # all_tokens = list(set(all_tokens))
        print("Total processed files : " + str(count_processed_files))
        return trees

    def _convert_ast_into_simpler_tree_format(self, root, binary_data):
        num_nodes = 0

        queue = [root]
        
        root_token = ""
        root_sub_tokens = []
        # Check if the node has children or not
        if len(root.children) == 0:
            root_token = binary_data[root_node.start_byte:root_node.end_byte]
            root_token_raw = root_token.decode("utf-8")
            root_token = self.process_token(root_token_raw)

            root_sub_tokens = split_identifier_into_parts(root_token_raw)
            root_sub_tokens = self.process_list_of_sub_tokens(root_sub_tokens)


        root_sub_token_ids = []
        for sub_token in root_sub_tokens:
            root_sub_token_ids.append(self.look_up_for_id_from_token(sub_token))

        root_json = {
            "node_type": str(root.type),
            "node_type_id": self.look_up_for_id_from_node_type(str(root.type)),
            "node_token": root_token,
            "node_sub_tokens": root_sub_tokens,
            "node_sub_tokens_id": root_sub_token_ids,
            "children": [] # Using children = None instead of [] to avoid the error 'Python 3: maximum recursion depth exceeded'
        }
      
        tree_tokens = []
        tree_tokens.extend(root_sub_tokens)
        
        queue_json = [root_json]
        while queue:
        
            current_node = queue.pop(0)
            current_node_json = queue_json.pop(0)
            num_nodes += 1

            children = [x for x in current_node.children]
            queue.extend(children)

            if len(children) > 0:
                current_node_json['children'] = []

            for child_node in children:

                child_token = ""
                child_sub_tokens = []
                if len(child_node.children) == 0:
                    child_token = binary_data[child_node.start_byte:child_node.end_byte]
                    child_token_raw = child_token.decode("utf-8")
                    child_token = self.process_token(child_token_raw)
                    child_sub_tokens = split_identifier_into_parts(str(child_token_raw))
                    child_sub_tokens = self.process_list_of_sub_tokens(child_sub_tokens)


                children_sub_token_ids = []
                for sub_token in child_sub_tokens:
                    sub_token = self.process_token(sub_token)
                    # print(sub_token)
                    sub_token_id = self.look_up_for_id_from_token(sub_token)
                    children_sub_token_ids.append(sub_token_id)

                if len(children_sub_token_ids) == 0:
                    children_sub_token_ids.append(0)

                child_json = {
                    "node_type": str(child_node.type),
                    "node_type_id": self.look_up_for_id_from_node_type(str(child_node.type)),
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

