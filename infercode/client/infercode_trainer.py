import sys
import os
import pickle
from pathlib import Path
sys.path.insert(0, '..')
import logging
from infercode.data_utils.ast_util import ASTUtil
from infercode.data_utils.token_vocab_extractor import TokenVocabExtractor
from infercode.data_utils.subtree_vocab_extractor import SubtreeVocabExtractor
from infercode.data_utils.dataset_processor import DatasetProcessor
from infercode.data_utils.threaded_iterator import ThreadedIterator
from infercode.data_utils.data_loader import DataLoader
from infercode.network.infercode_network import InferCodeModel
from infercode.data_utils.vocabulary import Vocabulary
from infercode.data_utils.language_util import LanguageUtil
import tensorflow.compat.v1 as tf
from .base_client import BaseClient
tf.disable_v2_behavior()

class InferCodeTrainer(BaseClient):

    LOGGER = logging.getLogger('InferCodeTrainer')

    def __init__(self, language):
        self.language = language


    def init_from_config(self, config=None):
        # Load default config if do not provide an external one
        self.load_configs(config)
        self.init_params()
        self.init_resources()
        self.init_utils()
        self.init_model_checkpoint()

        # ------------Set up the neural network------------
       
        self.training_data_processor = DatasetProcessor(input_data_path=self.data_path, 
                                               output_tensors_path=self.output_processed_data_path, 
                                               node_type_vocab_model_prefix=self.node_type_vocab_model_prefix, 
                                               node_token_vocab_model_prefix=self.node_token_vocab_model_prefix, 
                                               subtree_vocab_model_prefix=self.subtree_vocab_model_prefix, 
                                               language=self.language)
        self.training_buckets = self.training_data_processor.process_or_load_data()

        # self.ast_util, self.training_buckets = self.process_or_load_data()        
        self.data_loader = DataLoader(self.batch_size)
            
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
            self.LOGGER.error("Could not find the model : " + str(self.checkfile))
            self.LOGGER.error("Train the model from scratch")
        
        # -------------------------------------------------

    def train(self):
        for epoch in range(1,  self.epochs + 1):
            train_batch_iterator = ThreadedIterator(self.data_loader.make_minibatch_iterator(self.training_buckets), max_queue_size=5)
            for train_step, train_batch_data in enumerate(train_batch_iterator):
                _, err = self.sess.run(
                    [self.infercode_model.training_point,
                    self.infercode_model.loss],
                    feed_dict={
                        self.infercode_model.placeholders["node_type"]: train_batch_data["batch_node_type_id"],
                        self.infercode_model.placeholders["node_tokens"]:  train_batch_data["batch_node_tokens_id"],
                        self.infercode_model.placeholders["children_index"]:  train_batch_data["batch_children_index"],
                        self.infercode_model.placeholders["children_node_type"]: train_batch_data["batch_children_node_type_id"],
                        self.infercode_model.placeholders["children_node_tokens"]: train_batch_data["batch_children_node_tokens_id"],
                        self.infercode_model.placeholders["labels"]: train_batch_data["batch_subtree_id"],
                        self.infercode_model.placeholders["language_index"]: self.language_index,
                        self.infercode_model.placeholders["dropout_rate"]: 0.4
                    }
                )

                self.LOGGER.info(f"Training at epoch {epoch} and step {train_step} with loss {err}")
                v = train_step % self.checkpoint_every
                if train_step % self.checkpoint_every == 0:
                    self.saver.save(self.sess, self.checkfile)                  
                    self.LOGGER.info(f"Checkpoint saved, epoch {epoch} and step {train_step} with loss {err}")
