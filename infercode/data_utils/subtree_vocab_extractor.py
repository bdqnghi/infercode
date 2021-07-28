import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from .vocabulary import Vocabulary
from pathlib import Path
import os
from tqdm import *
from .subtree_util import SubtreeUtil
from .ast_parser import ASTParser
from .language_util import LanguageUtil
import os

class SubtreeVocabExtractor():


    def __init__(self, subtree_vocab_model_prefix: str):

        self.subtree_vocab_model_prefix = subtree_vocab_model_prefix
        self.subtree_vocab = Vocabulary(100000)
        self.subtree_util = SubtreeUtil()
        self.ast_parser = ASTParser()
        self.language_util = LanguageUtil()
        # self.ast_util = ASTUtil(node_type_vocab_model_path=node_type_vocab_model_path, 
        #                         node_token_vocab_model_path=node_token_vocab_model_path, language=language)

    def detect_language_of_file(self, file_path: str):
        _, file_extension = os.path.splitext(file_path)
        return self.language_util.get_language_by_file_extension(file_extension)

    def create_vocab_from_dir(self, input_data_path: str):
        all_subtrees_vocab = []
        for subdir , dirs, files in os.walk(input_data_path): 
            for file in tqdm(files):
                file_path = os.path.join(subdir, file)
                
                with open(file_path, "rb") as f:
                    code_snippet = f.read()

                language = self.detect_language_of_file(file_path)
                tree = self.ast_parser.parse(code_snippet, language)
                subtrees = self.subtree_util.extract_subtrees(tree)
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
        


          
      
