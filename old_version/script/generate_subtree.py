# select the language based on the existing shared objects
# parse the options to accept language and filename
# parse the code using tree-sitter of the correspnding language
# print a tree with depth, where node_id, node_type and node_label are separated by "-"
# print the subtree of a given node_id
# print all the subtrees of any node inside a tree
# print only the subtrees of the nodes of the selected node_types
# TODO assign unique IDs to node_types
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from utils.subtree_util import print_subtree
import argparse
from os.path import exists
import re
from os import path
from tree_sitter import Language, Parser
from pathlib import Path
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
parser.add_argument('--language', metavar='L', type=str, default="java", help='language to parse')
parser.add_argument('--filename', metavar='F', type=str, default="../examples/raw_code/104.c", help='file to parse')
#parser.add_argument('--node_types', type=str, help='a list of node types to be selected')
opt = parser.parse_args()

def main(opt):
    parser = Parser()
    print(opt.language[0])
    lang = Languages.get("java")
    parser.set_language(lang)
    lang_node_types_filename = "selected_node_types/{}.csv".format(opt.language[0])
    selected_node_types = {}
    if exists(lang_node_types_filename):
        lang_node_types = open(lang_node_types_filename, "r").read().splitlines()
        for lang_node_type in lang_node_types:
            selected_node_types[lang_node_type.lower()] = 1

    print("File name", opt.filename)
    data = open(opt.filename, "rb").read()
    tree = parser.parse(data)
    reports = {}
    s = print_subtree(data, tree.root_node, reports, selected_node_types)
    for report in reports:
        print("------------")
        print(reports[report])

if __name__ == "__main__":
    main(opt)
