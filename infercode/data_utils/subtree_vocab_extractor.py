import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from .vocabulary import Vocabulary
from pathlib import Path
import os
from tqdm import *
from .subtree_util import SubtreeUtil


class SubtreeVocabExtractor():


    def __init__(self, input_data_path: str, subtree_vocab_model_prefix: str,
                subtree_util: SubtreeUtil):

        self.input_data_path = input_data_path
        self.subtree_vocab_model_prefix = subtree_vocab_model_prefix
        self.subtree_vocab = Vocabulary(100000)
        self.subtree_util = subtree_util
        # self.ast_util = ASTUtil(node_type_vocab_model_path=node_type_vocab_model_path, 
        #                         node_token_vocab_model_path=node_token_vocab_model_path, language=language)

    def create_vocab(self):
        all_subtrees_vocab = []
        for subdir , dirs, files in os.walk(self.input_data_path): 
            for file in tqdm(files):
                file_path = os.path.join(subdir, file)
                
                with open(file_path, "rb") as f:
                    code_snippet = f.read()
                subtrees = self.subtree_util.extract_subtrees(code_snippet)
                all_subtrees_vocab.extend(subtrees)
        
        all_subtrees_vocab_filtered = []
        # Keep the subtrees with small size, ignore the large ones      
        for s in all_subtrees_vocab:
            if len(s) > 1 and len(s) < 8:
                all_subtrees_vocab_filtered.append(s)

        all_subtrees_vocab_concat = []        
        # Concat the list of nodes in a subtree into a string
        for s in all_subtrees_vocab_filtered:
            all_subtrees_vocab_concat.append("-".join(s))
        
        # model_type must be "word" for subtree vocab
        self.subtree_vocab.create_vocabulary(tokens=all_subtrees_vocab_concat, 
                                            model_filename=self.subtree_vocab_model_prefix, 
                                            model_type="word") 
        return self.subtree_vocab
        


          
      
