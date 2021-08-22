import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from data_utils.token_vocab_extractor import TokenVocabExtractor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--node_token_vocab_model_prefix', type=str, required=True)

def main(opt):
    

    token_vocab_extractor = TokenVocabExtractor(node_token_vocab_model_prefix=opt.node_token_vocab_model_prefix, 
                                                model_type="word")

    token_vocab_extractor.create_vocab_from_dir(opt.data_path)
# python3 extract_token_vocab.py --data_path ../../datasets/OJ_raw_small/ --node_token_vocab_model_prefix OJ_raw_token

if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)
