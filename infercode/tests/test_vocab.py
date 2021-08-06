import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from .data_utils.vocabulary import Vocabulary
import os
from dpu_utils.codeutils import identifiersplitting

nodes = []
with open("../tree-sitter-all-grammars-processed/node_types_all.csv","r") as f:
    data = f.readlines()
    for line in data:
        print(line)
        nodes.append(line.replace("\n",""))

print(nodes)
vocab = Vocabulary(5000)
vocab.create_vocabulary(tokens=nodes, model_filename="node_types_all", model_type="word")

# a = vocab.get_id_or_unk_for_text("for i in range(100, 2):")
# print(a)

# b = vocab.tokenize("for i in range(100, 2):")
# print(b)

# y = vocab.get_id_or_unk_for_text("do_statement")
# print(vocab.get_vocabulary())
