import os
import logging
import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from data_utils.ast_util import ASTUtil
import configparser
import tensorflow.compat.v1 as tf
from network.infercode_network import InferCodeModel
tf.disable_v2_behavior()

class InferCodeClient():

    LOGGER = logging.getLogger('InferCodeTrainer')

    def __init__(self, config):

        resource_config = config["resource"]
        nn_config = config["neural_network"]
        training_config = config["training_params"]

        self.node_type_vocab_model_prefix = resource_config["node_type_vocab_model_prefix"]
        self.node_token_vocab_model_prefix = resource_config["node_token_vocab_model_prefix"]
        self.subtree_vocab_model_prefix = resource_config["subtree_vocab_model_prefix"]
        self.language = resource_config["language"]

        self.ast_util = ASTUtil(node_type_vocab_model_path=self.node_type_vocab_model_prefix + ".model", 
                                node_token_vocab_model_path=self.node_token_vocab_model_prefix + ".model", language=self.language)
        
        self.model_checkpoint = training_config["model_checkpoint"]

        self.checkfile = os.path.join(self.model_checkpoint, 'cnn_tree.ckpt')
        ckpt = tf.train.get_checkpoint_state(self.model_checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            self.LOGGER.info("Load model successfully : " + str(self.checkfile))
        else:
            raise ValueError("Could not find the model : " + str(self.checkfile))

        self.infercode_model = InferCodeModel(config)

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def snippets_to_tensors(self, batch_code_snippets):
        batch_tree_indexes = []
        for code_snippet in batch_code_snippets:
            tree_representation, _ = self.ast_util.simplify_ast(str.encode(code_snippet))
            tree_indexes = self.ast_util.transform_tree_to_index(tree_representation)
            batch_tree_indexes.append(tree_indexes)
         
        tensors = self.ast_util.trees_to_batch_tensors(batch_tree_indexes)
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
                self.infercode_model.placeholders["dropout_rate"]: 0.0
            }
        )
        return embeddings[0]
