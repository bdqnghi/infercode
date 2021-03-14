import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from util.data.data_processor.treesitter_c_data_processor import TreeSitterDataProcessor


import argparse

def parse_arguments(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="../OJ_ctxl_train_test_val/train")
    parser.add_argument('--node_type_vocab_path',default="../vocab/ctxl/node_type/type.txt")
    parser.add_argument('--node_token_vocab_path', default="../vocab/ctxl/node_token/token.txt")
    parser.add_argument('--parser', type=str, default="ctxl", help="treesitter")
    opt = parser.parse_args(),
    return opt

def main(opt):
    data_path = opt.data_path
    node_type_vocab_path = opt.node_type_vocab_path
    node_token_vocab_path = opt.node_token_vocab_path
    parser = opt.parser
    
    
    if parser == "treesitter":
        processor = TreeSitterDataProcessor(node_type_vocab_path, node_token_vocab_path, data_path, parser)
   

if __name__ == "__main__":
    opt = parse_arguments()
    main(opt[0])
   