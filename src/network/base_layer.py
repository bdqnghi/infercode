import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))

class BaseLayer():
    def __init__(self, node_type_vocab_model_path: str, node_type_vocab_word_list_path: str,
                       token_vocab_model_path: str, token_vocab_word_list_path: str,
                       subtree_vocab_model_path: str, subtree_vocab_word_list_path: str):

        self.num_types = sum(1 for line in open(node_type_vocab_word_list_path))
        self.type_vocab = Vocabulary(num_type, node_type_vocab_model_path)

        self.num_tokens = sum(1 for line in open(token_vocab_word_list_path))
        self.token_vocab = Vocabulary(num_tokens, token_vocab_model_path)
        
        self.num_subtrees = sum(1 for line in open(subtree_vocab_word_list_path))
        self.subtree_vocab = Vocabulary(num_subtrees, subtree_vocab_model_path)

        self.batch_size = opt.batch_size

        self.placeholders = {}
        self.weights = {}
    
   