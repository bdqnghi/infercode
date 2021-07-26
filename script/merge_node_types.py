import os
import json

home = "../tree-sitter-all-grammars-processed"

all_nodes = []
for subdir , dirs, files in os.walk(home): 
    for file in files:
        file_path = os.path.join(subdir, file)
        nodes = open(file_path,"r").readlines()
        all_nodes.extend(nodes)

all_nodes = list(set(all_nodes))

with open("node_types_all.csv", "w") as f:
    for n in all_nodes:
        f.write(n.lower())
