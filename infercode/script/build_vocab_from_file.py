import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from data_utils.vocabulary import Vocabulary
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--text_file', type=str, required=True)
parser.add_argument('--model_file', type=str, required=True)
parser.add_argument('--model_type', type=str, default="word")

def main(opt):

    vocab = Vocabulary(10000000)
    vocab.create_vocabulary_from_file(sp_text_file=opt.text_file, model_filename=opt.model_file, model_type=opt.model_type)        

# python3 extract_token_vocab.py --data_path ../../datasets/OJ_raw_small/ --node_token_vocab_model_prefix OJ_raw_token

if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)
