import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from .vocabulary import Vocabulary
from pathlib import Path
import os
from tqdm import *

class SubtreeUtil():
    import logging
    LOGGER = logging.getLogger('SubtreeUtil')
    
    # def __init__(self, ast_parser):
        # self.ast_parser = ast_parser

    def extract_subtree(self, subtree_root):
        queue = [subtree_root]
        subtree_nodes = [subtree_root.type]
        ignore_types = ["\n"]
        while queue:
            
            current_node = queue.pop(0)
            for child in current_node.children:

                child_type = str(child.type)
                if child_type not in ignore_types:
                    queue.append(child)
                    subtree_nodes.append(child_type)
            
        return subtree_nodes

    def extract_subtrees(self, tree):
        # tree = self.ast_parser.parse(text, language)
        root = tree.root_node
        
        all_subtrees = []
        queue = [root]
        while queue:
            current_node = queue.pop(0)

            subtree = self.extract_subtree(current_node)
            all_subtrees.append(subtree)
            children = [x for x in current_node.children]
            queue.extend(children)
                
                # print(child_type)
        return all_subtrees
