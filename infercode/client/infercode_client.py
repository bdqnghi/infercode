import os
import logging
import coloredlogs
import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from infercode.data_utils.ast_util import ASTUtil
import configparser
import tensorflow.compat.v1 as tf
from infercode.network.infercode_network import InferCodeModel
from infercode.data_utils.vocabulary import Vocabulary
tf.disable_v2_behavior()

class InferCodeClient():

    LOGGER = logging.getLogger('InferCodeTrainer')

    def __init__(self, config):

        resource_config = config["resource"]
        training_config = config["training_params"]
        nn_config = config["neural_network"]

        self.data_path = resource_config["data_path"]
        self.output_processed_data_path = resource_config["output_processed_data_path"]
        self.node_type_vocab_model_prefix = resource_config["node_type_vocab_model_prefix"]
        self.node_token_vocab_model_prefix = resource_config["node_token_vocab_model_prefix"]
        self.subtree_vocab_model_prefix = resource_config["subtree_vocab_model_prefix"]
        self.language = resource_config["language"]

        # Training params
        self.model_checkpoint = training_config["model_checkpoint"]

        self.node_type_vocab = Vocabulary(100000, self.node_type_vocab_model_prefix + ".model")
        self.node_token_vocab = Vocabulary(100000, self.node_token_vocab_model_prefix + ".model")
        self.subtree_vocab = Vocabulary(100000, self.subtree_vocab_model_prefix + ".model")
        

        self.ast_util = ASTUtil(node_type_vocab_model_path=self.node_type_vocab_model_prefix + ".model", 
                                node_token_vocab_model_path=self.node_token_vocab_model_prefix + ".model", language=self.language)

        # ------------Set up the neural network------------
        self.infercode_model = InferCodeModel(num_types=self.node_type_vocab.get_vocabulary_size(), 
                                            num_tokens=self.node_token_vocab.get_vocabulary_size(), 
                                            num_subtrees=self.subtree_vocab.get_vocabulary_size(),
                                            num_conv=int(nn_config["num_conv"]), 
                                            node_type_dim=int(nn_config["node_type_dim"]), 
                                            node_token_dim=int(nn_config["node_token_dim"]),
                                            conv_output_dim=int(nn_config["conv_output_dim"]), 
                                            include_token=int(nn_config["include_token"]), 
                                            batch_size=int(nn_config["batch_size"]), 
                                            learning_rate=float(nn_config["lr"]))

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
            raise ValueError("Could not find the model : " + str(self.checkfile))

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
