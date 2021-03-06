import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from .vocabulary import Vocabulary
from pathlib import Path
import os
from tqdm import *
from .ast_util import ASTUtil


class SubtreeVocabExtractor():


    def __init__(self, input_data_path: str, output_subtree_vocab_prefix: str,
                node_type_vocab_model_path: str, node_token_vocab_model_path: str, ast_util: ASTUtil):

        self.input_data_path = input_data_path
        self.output_subtree_vocab_prefix = output_subtree_vocab_prefix
        self.subtree_vocab = Vocabulary(100000)
        self.ast_util = ast_util
        # self.ast_util = ASTUtil(node_type_vocab_model_path=node_type_vocab_model_path, 
        #                         node_token_vocab_model_path=node_token_vocab_model_path, language=language)

    def create_vocab(self):
        all_subtrees_vocab = []
        for subdir , dirs, files in os.walk(self.input_data_path): 
            for file in tqdm(files):
                file_path = os.path.join(subdir, file)
                
                with open(file_path, "rb") as f:
                    code_snippet = f.read()
                subtrees = self.ast_util.extract_subtrees(code_snippet)
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
                                            model_filename=self.output_subtree_vocab_prefix, 
                                            model_type="word") 
        return self.subtree_vocab
        


          
      
