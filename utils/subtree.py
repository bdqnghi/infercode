oimport argparse
from os.path import exists

# select the language based on the existing shared objects
# parse the options to accept language and filename
# parse the code using tree-sitter of the correspnding language
# print a tree with depth, where node_id, node_type and node_label are separated by "-"
# print the subtree of a given node_id
# print all the subtrees of any node inside a tree
# print only the subtrees of the nodes of the selected node_types
# TODO assign unique IDs to node_types

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
parser.add_argument('language', metavar='L', type=str, nargs=1, help='language to parse')
parser.add_argument('filename', metavar='F', type=str, nargs=1, help='file to parse')
#parser.add_argument('--node_types', type=str, help='a list of node types to be selected')
opt = parser.parse_args()

def print_tree_line(id, data, root_node, reports, selected_node_types):
        node_id = id
        node_type = root_node.type
        node_label = data[root_node.start_byte:root_node.end_byte]
        has_child = len(root_node.children) > 0
        depth = 1
        s = "{}-{},".format(node_id, node_type)
        if not has_child:
            s = "{}-{}-{},".format(node_id, node_type, node_label.decode("utf-8"))
        for child in root_node.children:
            (id, child_depth, child_str) = print_tree_line(id + 1, data, child, reports, selected_node_types)
            depth = max(depth, child_depth+1)
            s = "{}{}".format(s, child_str)
        if str(node_type) in selected_node_types:
           reports[node_id] = "{}{}".format(s, depth)
        return (id, depth, s)

def print_subtree(data, root_node, reports, selected_node_types):
        (id, depth, s) = print_tree_line(1, data, root_node, reports, selected_node_types)
        return "{}{}".format(s, depth)

def main(opt):
        parser = Parser()
        lang = Languages.get(opt.language[0])
        parser.set_language(lang)
        lang_node_types_filename = "node_types_{}.csv".format(opt.language[0])
        selected_node_types = {}
        if exists(lang_node_types_filename):
            lang_node_types = open(lang_node_types_filename, "r").read().splitlines()
            for lang_node_type in lang_node_types:
                selected_node_types[lang_node_type.lower()] = 1
        data = open(opt.filename[0], "rb").read()
        tree = parser.parse(data)
        reports = {}
        s = print_subtree(data, tree.root_node, reports, selected_node_types)
        for report in reports:
            print(reports[report])

if __name__ == "__main__":
        main(opt)
