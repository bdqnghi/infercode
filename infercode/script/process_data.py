import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from data_utils.dataset_processor import DatasetProcessor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--output_processed_data_path', type=str, required=True)
parser.add_argument('--node_type_vocab_model_prefix', type=str, required=True)
parser.add_argument('--node_token_vocab_model_prefix', type=str, required=True)
parser.add_argument('--subtree_vocab_model_prefix', type=str, required=True)
parser.add_argument('--language', type=str)

def main(opt):
  
    data_processor = DatasetProcessor(input_data_path=opt.data_path, 
                                      output_tensors_path=opt.output_processed_data_path, 
                                      node_type_vocab_model_prefix=opt.node_type_vocab_model_prefix, 
                                      node_token_vocab_model_prefix=opt.node_token_vocab_model_prefix, 
                                      subtree_vocab_model_prefix=opt.subtree_vocab_model_prefix, 
                                      language=opt.language)

    data_processor.process_or_load_data()
# python3 process_data.py --data_path ../../datasets/OJ_raw_small/ --output_processed_data_path ../../datasets/OJ_raw_processed/OJ_raw_small.pkl --node_type_vocab_model_prefix ../../sentencepiece_vocab/node_types/node_types_all --node_token_vocab_model_prefix ../../sentencepiece_vocab/tokens/OJ_raw_bpe --subtree_vocab_model_prefix ../../sentencepiece_vocab/subtrees/OJ_raw_subtree --language c

if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)