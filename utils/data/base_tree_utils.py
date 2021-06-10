from bidict import bidict
import pickle

class BaseTreeUtils():

    def __init__(self, opt):

        self.node_type_lookup = self.load_node_type_vocab(opt.node_type_vocabulary_path)
        self.node_token_lookup = self.load_node_token_vocab(opt.token_vocabulary_path)
        self.subtree_lookup = self.load_subtree_vocab(opt.subtree_vocabulary_path)

    def look_up_for_id_of_token(self, token):
        
        token_id = self.node_token_lookup["<SPECIAL>"]
        if token in self.node_token_lookup:
            token_id = self.node_token_lookup[token]

        return token_id

    def look_up_for_token_of_id(self, token_id):
        token = self.node_token_lookup.inverse[token_id]
        return token

    def load_tree_from_pickle_file(self, file_path):
        """Builds an AST from a script."""
   
        with open(file_path, 'rb') as file_handler:
            tree = pickle.load(file_handler)
            # print(tree)
            return tree
        return "error"


    def load_node_type_vocab(self, node_type_vocabulary_path):

        node_type_lookup = {}
       
        with open(node_type_vocabulary_path, "r") as f2:
            data = f2.readlines()
            for line in data:
                splits = line.replace("\n", "").split(",")
                node_type_lookup[splits[1]] = int(splits[0])

        node_type_lookup = bidict(node_type_lookup)
        return node_type_lookup

    def load_node_token_vocab(self, token_vocabulary_path):
        node_token_lookup = {}
        with open(token_vocabulary_path, "r") as f3:
            data = f3.readlines()
            for line in data:
                splits = line.replace("\n", "").split(",")
                node_token_lookup[splits[1]] = int(splits[0])

        node_token_lookup = bidict(node_token_lookup)
        return node_token_lookup

    def load_subtree_vocab(self, subtree_vocabulary_path):
        subtree_lookup = {}
        with open(subtree_vocabulary_path, "r") as f4:
            data = f4.readlines()
            for i, line in enumerate(data):
                splits = line.replace("\n", "").split(",")
                subtree_lookup[splits[0]] = i

        subtree_lookup = bidict(subtree_lookup)
        return subtree_lookup