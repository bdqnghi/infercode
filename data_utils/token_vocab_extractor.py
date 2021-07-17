from bidict import bidict

from dpu_utils.codeutils import identifiersplitting
import pickle
import sys
from tqdm import trange
from tqdm import *
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from .vocabulary import Vocabulary
import os

class TokenVocabExtractor():

    def __init__(self, input_data_path: str, node_type_vocab_path: str="java-small", model_type: str="bpe") -> None:
        self.input_data_path = input_data_path
        self.node_type_vocab_path = node_type_vocab_path
        self.model_type = model_type
        self.token_vocab = Vocabulary(100000)

    def create_vocab(self) -> None:
        all_tokens = []
        for subdir , dirs, files in os.walk(self.input_data_path): 
            for file in tqdm(files):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r") as f:
                    data = str(f.read())
                    data = data.replace("\n", "")
                    tokens = identifiersplitting.split_identifier_into_parts(data)
                    all_tokens.append(tokens)
        self.token_vocab.create_vocabulary(tokens=all_tokens, model_filename=self.node_type_vocab_path, model_type=self.model_type)        