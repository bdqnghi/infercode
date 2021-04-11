import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from utils.data.data_processor.treesitter_data_processor import TreeSitterDataProcessor


import argparse

def parse_arguments(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="../java-small-test/training")
    parser.add_argument('--node_type_vocab_path',default="../vocab/treesitter/node_type/node_types_c_java_cpp_c-sharp_rust.txt")
    parser.add_argument('--node_token_vocab_path', default="../vocab/treesitter/node_token/token.txt")
    parser.add_argument('--subtree_vocab_path', default="../subtrees/subtrees.txt")
    # parser.add_argument('--parser', type=str, default="treesitter", help="treesitter")
    opt = parser.parse_args(),
    return opt

def main(opt):
    data_path = opt.data_path
    node_type_vocab_path = opt.node_type_vocab_path
    node_token_vocab_path = opt.node_token_vocab_path
    subtree_vocab_path = opt.subtree_vocab_path
    # parser = opt.parser
    
    
    # if parser == "treesitter":
    processor = TreeSitterDataProcessor(node_type_vocab_path, node_token_vocab_path, subtree_vocab_path, data_path)
   

if __name__ == "__main__":
    opt = parse_arguments()
    main(opt[0])
   