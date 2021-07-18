from dpu_utils.codeutils import identifiersplitting
import sys
from tqdm import *
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from .vocabulary import Vocabulary
import os

class TokenVocabExtractor():

    def __init__(self, input_data_path: str, node_token_vocab_prefix: str, model_type: str="bpe") -> None:
        self.input_data_path = input_data_path
        self.node_token_vocab_prefix = node_token_vocab_prefix
        self.model_type = model_type
        self.token_vocab = Vocabulary(100000)

    def create_vocab(self):
        all_tokens = []
        for subdir , dirs, files in os.walk(self.input_data_path): 
            for file in tqdm(files):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r") as f:
                    data = str(f.read())
                    data = data.replace("\n", "")
                    tokens = identifiersplitting.split_identifier_into_parts(data)
                    all_tokens.append(tokens)
        self.token_vocab.create_vocabulary(tokens=all_tokens, model_filename=self.node_token_vocab_prefix, model_type=self.model_type)        
        return self.token_vocab