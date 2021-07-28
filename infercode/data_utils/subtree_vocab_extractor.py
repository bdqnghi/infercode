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
        self.temp_subtrees_file = "temp_subtrees.csv"
        if os.path.exists(self.temp_subtrees_file):
            os.remove(self.temp_subtrees_file)

    def detect_language_of_file(self, file_path: str):
        _, file_extension = os.path.splitext(file_path)
        return self.language_util.get_language_by_file_extension(file_extension)

    def create_vocab_from_dir(self, input_data_path: str):
        for subdir , dirs, files in os.walk(input_data_path): 
            for file in tqdm(files):
                file_path = os.path.join(subdir, file)
                
                with open(file_path, "rb") as f:
                    code_snippet = f.read()

                language = self.detect_language_of_file(file_path)
                tree = self.ast_parser.parse(code_snippet, language)
                subtrees = self.subtree_util.extract_subtrees(tree)
                # Keep the subtrees with small size, ignore the large ones      
                for s in subtrees:
                    if len(s) > 1 and len(s) < 8:
                        # Concat the list of nodes in a subtree into a string
                        subtree_str = "-".join(s)
                        if subtree_str not in all_subtrees_vocab:
                            # Write to a temporary file as keeping a large array may cause memory overflow
                            with open(self.temp_subtrees_file, "a") as f:
                                # all_subtrees_vocab.append(subtree_str)
                                f.write(subtree_str)
                                f.write("\n")
        
        # all_subtrees_vocab = []
        with open(self.temp_subtrees_file, "r") as f1:
            all_subtrees_vocab = f1.read().splitlines()

        # model_type must be "word" for subtree vocab
        self.subtree_vocab.create_vocabulary(tokens=all_subtrees_vocab, 
                                            model_filename=self.subtree_vocab_model_prefix, 
                                            model_type="word") 
        return self.subtree_vocab
        


          
      
