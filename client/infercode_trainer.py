import sys
import os
import pickle
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
import logging
from data_utils.ast_util import ASTUtil
from data_utils.token_vocab_extractor import TokenVocabExtractor
from data_utils.subtree_vocab_extractor import SubtreeVocabExtractor
from data_utils.dataset_processor import DatasetProcessor
from data_utils.threaded_iterator import ThreadedIterator
from data_utils.data_loader import DataLoader
from network.infercode_network import InferCodeModel
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
logging.basicConfig(level=logging.INFO)

class InferCodeTrainer():

    LOGGER = logging.getLogger('InferCodeTrainer')

    def __init__(self, config):
        
        resource_config = config["resource"]
        training_config = config["training_params"]

        self.data_path = resource_config["data_path"]
        self.output_processed_data_path = resource_config["output_processed_data_path"]
        self.node_type_vocab_model_prefix = resource_config["node_type_vocab_model_prefix"]
        self.node_token_vocab_model_prefix = resource_config["node_token_vocab_model_prefix"]
        self.subtree_vocab_model_prefix = resource_config["subtree_vocab_model_prefix"]
        self.language = resource_config["language"]

        # Training params
        self.epochs = int(training_config["epochs"])
        self.batch_size = int(training_config["batch_size"])
        self.checkpoint_every = int(training_config["checkpoint_every"])
        self.model_checkpoint = training_config["model_checkpoint"]

        self.ast_util, self.training_buckets = self.process_or_load_data()        
        self.data_loader = DataLoader(self.ast_util, self.batch_size)

        # ------------Set up the neural network------------
        self.infercode_model = InferCodeModel(config)
        self.saver = tf.train.Saver(save_relative_paths=True, max_to_keep=5)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        # -------------------------------------------------

    def process_or_load_data(self):
        """
        Generate the code token vocabulary, the token will be tokenized using the 
        Byte-Pair Encoding using sentencepiece from https://github.com/google/sentencepiece.
        There will be 2 outputs:
            - Model: node_token_vocab_model_prefix + ".model".
            - Vocab list: node_token_vocab_model_prefix + ".vocab".
        """
        if not os.path.exists(self.node_token_vocab_model_prefix + ".model"):
            self.LOGGER.info("Generating token vocabulary")
            token_vocab_extractor = TokenVocabExtractor(input_data_path=self.data_path, 
                                                        node_token_vocab_prefix=self.node_token_vocab_model_prefix, 
                                                        model_type="bpe")
            token_vocab_extractor.create_vocab()
       
        
        """
        Initialize the ast utility, it requires the token vocabulary as the input.
        """
        self.LOGGER.info("Initializing AST Utilities")
        ast_util = ASTUtil(node_type_vocab_model_path=self.node_type_vocab_model_prefix + ".model", 
                                node_token_vocab_model_path=self.node_token_vocab_model_prefix + ".model", language=self.language)


        """
        Generate the subtree vocabulary.
        There will be 2 outputs:
            - Model: subtree_vocab_model_prefix + ".model".
            - Vocab list: subtree_vocab_model_prefix + ".vocab".
        """
        if not os.path.exists(self.subtree_vocab_model_prefix + ".model"):
            self.LOGGER.info("Generating subtree vocabulary")
            subtree_vocab_extractor = SubtreeVocabExtractor(input_data_path=self.data_path, output_subtree_vocab_prefix=self.subtree_vocab_model_prefix,
                                                            node_type_vocab_model_path=self.node_type_vocab_model_prefix + ".model", 
                                                            node_token_vocab_model_path=self.node_token_vocab_model_prefix + ".model",
                                                            ast_util=self.ast_util)
            subtree_vocab = subtree_vocab_extractor.create_vocab()

        """
        Pre-process the data after generating the token vocab and the subtree vocab.
        Steps including:
            - Walk through each file (supposed to be the code snippet) in the dataset.
            - Parse the code snippet into AST.
            - Simplify the AST
            - Transform the simplied AST into indexes.
            - Extract subtrees from the AST.
            - Put trees with similar size into the same bucket
        The output will be:
            - output_processed_data_path: a pickle file that contains the processed data
        """
        if not os.path.exists(self.output_processed_data_path):
            self.LOGGER.info("Processing the dataset")
            dataset_processor = DatasetProcessor(input_data_path=self.data_path, output_tensors_path=self.output_processed_data_path,
                                                 node_type_vocab_model_path=self.node_type_vocab_model_prefix + ".model", 
                                                 node_token_vocab_model_path=self.node_token_vocab_model_prefix + ".model",
                                                 subtree_vocab_model_path=self.subtree_vocab_model_prefix + ".model",
                                                 ast_util=self.ast_util)
            training_buckets = dataset_processor.process()
        else:
            training_buckets = pickle.load(open(self.output_processed_data_path, "rb"))

        return ast_util, training_buckets


    def train(self):
        for epoch in range(1,  self.epochs + 1):
            train_batch_iterator = ThreadedIterator(self.data_loader.make_minibatch_iterator(self.training_buckets), max_queue_size=1)
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
                        self.infercode_model.placeholders["dropout_rate"]: 0.4
                    }
                )

                self.LOGGER.info(f"Training at epoch {epoch} and step {train_step} with loss {err}")
                if train_step % self.checkpoint_every == 100 and train_step > 0:
                    self.saver.save(sess, checkfile)                  
                    print('Checkpoint saved, epoch:' + str(epoch) + ', step: ' + str(train_step) + ', loss: ' + str(err) + '.')
