import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from utils.subtree_util import print_subtree
import argparse
from os.path import exists
import re
from os import path
from tree_sitter import Language, Parser
import tqdm
from dpu_utils.codeutils import split_identifier_into_parts

home = str(Path.home())

import glob, os
cd = os.getcwd()
os.chdir(path.join(home, ".tree-sitter", "bin"))
Languages = {}
for file in glob.glob("*.so"):
  try:
    lang = os.path.splitext(file)[0]
    Languages[lang] = Language(path.join(home, ".tree-sitter", "bin", file), lang)
  except:
    print("An exception occurred to {}".format(lang))
os.chdir(cd)

def look_up_for_id_from_node_type(type):
    return 1

def look_up_for_id_from_token(type):
    return 2

def process_token(text):
    return text

def _convert_ast_into_simpler_tree_format(root, binary_data):
    num_nodes = 0

    queue = [root]
    
    root_token = ""
    # Check if the node has children or not
    if len(root.children) == 0:
        root_token = binary_data[root_node.start_byte:root_node.end_byte].lower()
        root_token = root_token.decode("utf-8")
        root_token = process_token(root_token)

    root_sub_tokens = split_identifier_into_parts(root_token)
    
    root_sub_token_ids = []
    for sub_token in root_sub_tokens:
        root_sub_token_ids.append(look_up_for_id_from_token(sub_token))

    root_json = {
        "node_type": str(root.type),
        "node_type_id": look_up_for_id_from_node_type(str(root.type)),
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
            if len(child_node.children) == 0:
                child_token = binary_data[child_node.start_byte:child_node.end_byte].lower()
                child_token = child_token.decode("utf-8")
                print(child_token)
                child_token = process_token(child_token)

            child_sub_tokens = split_identifier_into_parts(str(child_token))

            children_sub_token_ids = []
            for sub_token in child_sub_tokens:
                sub_token = process_token(sub_token)
                # print(sub_token)
                sub_token_id = look_up_for_id_from_token(sub_token)
                children_sub_token_ids.append(sub_token_id)

            if len(children_sub_token_ids) == 0:
                children_sub_token_ids.append(0)

            child_json = {
                "node_type": str(child_node.type),
                "node_type_id": look_up_for_id_from_node_type(str(child_node.type)),
                "node_token": child_token,
                "node_sub_tokens": child_sub_tokens,
                "node_sub_tokens_id": children_sub_token_ids,
                "children": []
            }

            tree_tokens.extend(child_sub_tokens)

            current_node_json['children'].append(child_json)
            queue_json.append(child_json)

    tree_tokens = list(set(tree_tokens))

    print(root_json)
    print(num_nodes)
    return root_json, tree_tokens, num_nodes

path = "examples/raw_code/104.c"
parser = Parser()
parser_lang = Languages.get("java")
parser.set_language(parser_lang)

data = open(path, "rb").read()
tree = parser.parse(data)
_convert_ast_into_simpler_tree_format(tree.root_node, data)