import os
import logging
import coloredlogs
import sys
sys.path.insert(0, '..')
from infercode.data_utils.ast_util import ASTUtil
from infercode.data_utils.ast_parser import ASTParser
import configparser
import tensorflow.compat.v1 as tf
from infercode.network.infercode_network import InferCodeModel
from infercode.data_utils.vocabulary import Vocabulary
from infercode.data_utils.language_util import LanguageUtil
from infercode.data_utils.tensor_util import TensorUtil
from .base_client import BaseClient
tf.disable_v2_behavior()

class InferCodeClient(BaseClient):

    LOGGER = logging.getLogger('InferCodeTrainer')

    def __init__(self, language):

        self.language = language

    def init_from_config(self, config=None):        
        
        self.load_configs(config)
        self.init_params()
        self.init_resources()
        self.init_utils()
        self.init_model_checkpoint()

        # ------------Set up the neural network------------
        self.infercode_model = InferCodeModel(num_types=self.node_type_vocab.get_vocabulary_size(), 
                                              num_tokens=self.node_token_vocab.get_vocabulary_size(), 
                                              num_subtrees=self.subtree_vocab.get_vocabulary_size(),
                                              num_languages=self.language_util.get_num_languages(),
                                              num_conv=self.num_conv, 
                                              node_type_dim=self.node_type_dim, 
                                              node_token_dim=self.node_token_dim,
                                              conv_output_dim=self.conv_output_dim, 
                                              include_token=self.include_token, 
                                              batch_size=self.batch_size, 
                                              learning_rate=self.learning_rate)

        self.saver = tf.train.Saver(save_relative_paths=True, max_to_keep=5)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.checkfile = os.path.join(self.model_checkpoint, 'cnn_tree.ckpt')
        ckpt = tf.train.get_checkpoint_state(self.model_checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            self.LOGGER.info("Load model successfully : " + str(self.checkfile))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            error_message = "Could not find the model : " + str(self.checkfile)
            self.LOGGER.error(error_message)
            raise ValueError(error_message)

    def snippets_to_tensors(self, batch_code_snippets):
        batch_tree_indexes = []
        for code_snippet in batch_code_snippets:
            ast = self.ast_parser.parse(str.encode(code_snippet))
            tree_representation, _ = self.ast_util.simplify_ast(ast, str.encode(code_snippet))
            tree_indexes = self.tensor_util.transform_tree_to_index(tree_representation)
            batch_tree_indexes.append(tree_indexes)
         
        tensors = self.tensor_util.trees_to_batch_tensors(batch_tree_indexes)
        return tensors


    def encode(self, batch_code_snippets):
        tensors = self.snippets_to_tensors(batch_code_snippets)
        embeddings = self.sess.run(
            [self.infercode_model.code_vector],
            feed_dict={
                self.infercode_model.placeholders["node_type"]: tensors["batch_node_type_id"],
                self.infercode_model.placeholders["node_tokens"]:  tensors["batch_node_tokens_id"],
                self.infercode_model.placeholders["children_index"]:  tensors["batch_children_index"],
                self.infercode_model.placeholders["children_node_type"]: tensors["batch_children_node_type_id"],
                self.infercode_model.placeholders["children_node_tokens"]: tensors["batch_children_node_tokens_id"],
                self.infercode_model.placeholders["language_index"]: self.language_index,
                self.infercode_model.placeholders["dropout_rate"]: 0.0
            }
        )
        print(embeddings[0].shape)
        return embeddings[0]
