import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from data_utils.subtree_vocab_extractor import SubtreeVocabExtractor
from data_utils.ast_parser import ASTParser
from data_utils.subtree_util import SubtreeUtil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--subtree_vocab_model_prefix', type=str, required=True)
parser.add_argument('--language', type=str, required=True)

def main(opt):
    
    ast_parser = ASTParser(language=opt.language)
    subtree_util = SubtreeUtil(ast_parser=ast_parser)

    subtree_vocab_extractor = SubtreeVocabExtractor(input_data_path=opt.data_path, 
                                                    subtree_vocab_model_prefix=opt.subtree_vocab_model_prefix, 
                                                    subtree_util=subtree_util)

    subtree_vocab_extractor.create_vocab()
# python3 extract_subtree_vocab.py --data_path ../../datasets/OJ_raw_small/ --output_subtree_vocab_prefix OJ_raw_subtree --language c

if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)