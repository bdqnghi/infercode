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

parser = argparse.ArgumentParser()
parser.add_argument('--input_directory', metavar='F', type=str, default="../java-small-test", help='file to parse')
parser.add_argument('--output_file', metavar='F', type=str, default="../examples/raw_code/104.c", help='file to parse')
parser.add_argument('--node_type_path', type=str, default="../grammars/node_types_c_java_cpp_c-sharp.csv", help='path to node types')
opt = parser.parse_args()

def load_node_types_vocab(path):
    node_types_dict = {}

    node_types = open(path).read().splitlines()
    for i, n in enumerate(node_types):
        node_types_dict[n] = i

    return node_types_dict


def process_subtree_line(subtree_line, node_types_dict):
    
    line_splits = subtree_line.split(",")

    nodes = line_splits[:-1]
    depth = int(line_splits[-1])

    node_type_indices = []

    if depth > 0 and depth < 6:
        for node in nodes:
          
            try:
                node_splits = node.split("-")
                node_id = node_splits[0]
                node_type = node_splits[1]

                if node_type in node_types_dict:
                    node_type_index = node_types_dict[node_type]
                    node_type_indices.append(str(node_type_index))
            except Exception as e:
                print(e)

    return node_type_indices

def main(opt):

    node_types_vocab_path = opt.node_type_path
    node_types_dict = load_node_types_vocab(node_types_vocab_path)

    parser = Parser()

    excluded_node_types = open("excluded_node_types/list.csv").read().splitlines()
    # print(excluded_node_types)

    subtrees = []
    for subdir , dirs, files in os.walk(opt.input_directory): 
        for file in files:
            file_path = os.path.join(subdir,file)
            print(file_path)
            lang = "java"
            if file.endswith("c"):
                lang = "c"
            if file.endswith("cpp"):
                lang = "cpp"
            if file.endswith("cs"):
                lang = "c_sharp"
            if file.endswith("rs"):
                lang = "rust"
            if file.endswith("go"):
                lang = "go"

            parser_lang = Languages.get(lang)
            parser.set_language(parser_lang)

            data = open(file_path, "rb").read()
            tree = parser.parse(data)
            subtrees_flattened = {}
            print_subtree(data, tree.root_node, subtrees_flattened, excluded_node_types)

            # print(subtrees_flattened)
            
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

    subtrees = list(set(subtrees))
    subtrees.sort(key=len)
    for s in subtrees:
        with open("../subtrees/subtrees.txt", "a") as f:
            f.write(s)
            f.write("\n")

if __name__ == "__main__":
    main(opt)