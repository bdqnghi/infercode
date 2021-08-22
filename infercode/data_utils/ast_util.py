import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from os import path
from .vocabulary import Vocabulary
from tree_sitter import Language, Parser
from pathlib import Path
import glob, os
import numpy as np
import logging
from dpu_utils.codeutils import identifiersplitting


class ASTUtil():
    import logging
    LOGGER = logging.getLogger('ASTUtil')

    def __init__(self, node_type_vocab_model_path: str, node_token_vocab_model_path: str):

        self.type_vocab = Vocabulary(1000, node_type_vocab_model_path)
        self.token_vocab = Vocabulary(100000, node_token_vocab_model_path)

    # Simplify the AST 
    def simplify_ast(self, tree, text):
        # tree = self.ast_parser.parse(text)
        root = tree.root_node

        ignore_types = ["\n"]
        num_nodes = 0
        root_type = str(root.type)
        root_type_id = self.type_vocab.get_id_or_unk_for_text(root_type)[0]
        queue = [root]

        root_json = {
            "node_type": root_type,
            "node_type_id": root_type_id,
            "node_tokens": [],
            "node_tokens_id": [],
            "children": []
        }

        queue_json = [root_json]
        while queue:
            
            current_node = queue.pop(0)
            current_node_json = queue_json.pop(0)
            num_nodes += 1


            for child in current_node.children:
                child_type = str(child.type)
                if child_type not in ignore_types:
                    queue.append(child)

                    
                    child_type_id = self.type_vocab.get_id_or_unk_for_text(child_type)[0]

                    child_token = ""
                    child_sub_tokens_id = []
                    child_sub_tokens = []

                    has_child = len(child.children) > 0

                    if not has_child:
                        child_token = text[child.start_byte:child.end_byte].decode("utf-8")
                        child_sub_tokens_id = self.token_vocab.get_id_or_unk_for_text(child_token)
                        subtokens = " ".join(identifiersplitting.split_identifier_into_parts(child_token))
                        child_sub_tokens = self.token_vocab.tokenize(subtokens)

                    if len(child_sub_tokens_id) == 0:
                        child_sub_tokens_id.append(0)
       
                    # print(children_sub_token_ids)
                    child_json = {
                        "node_type": child_type,
                        "node_type_id": child_type_id,
                        "node_tokens": child_sub_tokens,
                        "node_tokens_id":child_sub_tokens_id,
                        "children": []
                    }

                    current_node_json['children'].append(child_json)
                    queue_json.append(child_json)

        return root_json, num_nodes

   