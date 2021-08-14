from dpu_utils.codeutils import identifiersplitting
import sys
from tqdm import *
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from .vocabulary import Vocabulary
import os

class TokenVocabExtractor():

    def __init__(self, node_token_vocab_model_prefix: str, model_type: str="bpe") -> None:
        self.node_token_vocab_model_prefix = node_token_vocab_model_prefix
        self.model_type = model_type
        self.token_vocab = Vocabulary(100000)

    def create_vocab_from_dir(self, input_data_path: str):
        all_tokens = []
        for subdir , dirs, files in os.walk(input_data_path): 
            for file in tqdm(files):
                # if file.endswith(file_types):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r", errors='ignore') as f:
                    data = str(f.read())
                    data = data.replace("\n", " ")
                    tokens = identifiersplitting.split_identifier_into_parts(data)
                    all_tokens.extend(tokens)
        self.token_vocab.create_vocabulary(tokens=all_tokens, model_filename=self.node_token_vocab_model_prefix, model_type=self.model_type)        
        return self.token_vocab