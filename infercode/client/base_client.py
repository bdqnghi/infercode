import os
import logging
import coloredlogs
import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from data_utils.ast_util import ASTUtil
from data_utils.ast_parser import ASTParser
import configparser
import tensorflow.compat.v1 as tf
from network.infercode_network import InferCodeModel
from data_utils.vocabulary import Vocabulary
from data_utils.language_util import LanguageUtil
from data_utils.tensor_util import TensorUtil
tf.disable_v2_behavior()

class BaseClient():

    LOGGER = logging.getLogger('BaseClient')

    def __init__(self, language):

        self.language = language

    def init_from_config(self, config=None):        
        
        self.init_params(config)


    def init_params(self, config=None):
        if config == None:
            import configparser 
            current_path = os.path.dirname(os.path.realpath(__file__))
            current_path = Path(current_path)
            parent_of_current_path = current_path.parent.absolute()
            config = configparser.ConfigParser()
            default_config_path = os.path.join(parent_of_current_path, "configs/default_config.ini")
            config.read(default_config_path)

        resource_config = config["resource"]
        training_config = config["training_params"]
        nn_config = config["neural_network"]

        self.data_path = resource_config["data_path"]
        self.output_processed_data_path = resource_config["output_processed_data_path"]
        self.node_type_vocab_model_prefix = resource_config["node_type_vocab_model_prefix"]
        self.node_token_vocab_model_prefix = resource_config["node_token_vocab_model_prefix"]
        self.subtree_vocab_model_prefix = resource_config["subtree_vocab_model_prefix"]

        # Training params
        self.epochs = int(training_config["epochs"])
        self.batch_size = int(nn_config["batch_size"])
        self.checkpoint_every = int(training_config["checkpoint_every"])
        self.model_checkpoint = training_config["model_checkpoint"]

    def initialize_model_checkpoint(sefl):


    def init_utils(self):

        self.node_type_vocab = Vocabulary(100000, self.node_type_vocab_model_prefix + ".model")
        self.node_token_vocab = Vocabulary(100000, self.node_token_vocab_model_prefix + ".model")
        self.subtree_vocab = Vocabulary(100000, self.subtree_vocab_model_prefix + ".model")

        self.ast_parser = ASTParser(self.language)
        self.ast_util = ASTUtil(node_type_vocab_model_path=self.node_type_vocab_model_prefix + ".model", 
                                node_token_vocab_model_path=self.node_token_vocab_model_prefix + ".model")

        self.language_util = LanguageUtil()
        self.language_index = self.language_util.get_language_index(self.language)
        
        self.tensor_util = TensorUtil()