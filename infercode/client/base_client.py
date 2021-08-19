import os
import logging
from pathlib import Path
import sys
from infercode.data_utils.ast_util import ASTUtil
from infercode.data_utils.ast_parser import ASTParser
import configparser
import tensorflow.compat.v1 as tf
from infercode.network.infercode_network import InferCodeModel
from infercode.data_utils.vocabulary import Vocabulary
from infercode.data_utils.language_util import LanguageUtil
from infercode.data_utils.tensor_util import TensorUtil
from os import path
import pathlib
tf.disable_v2_behavior()
import urllib.request
from tqdm import tqdm
import zipfile
import os
package_dir = Path(__file__).parents[1]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class BaseClient():

    LOGGER = logging.getLogger('BaseClient')

    def __init__(self, language):

        self.language = language


    def load_configs(self, config=None):
        if config == None:
            import configparser 
            current_path = os.path.dirname(os.path.realpath(__file__))
            current_path = Path(current_path)
            parent_of_current_path = current_path.parent.absolute()
            config = configparser.ConfigParser()
            default_config_path = os.path.join(parent_of_current_path, "configs/default_config.ini")
            config.read(default_config_path)

        self.resource_config = config["resource"]
        self.training_config = config["training_params"]
        self.nn_config = config["neural_network"]

    
    def init_params(self):
        # Training params
        self.epochs = int(self.training_config["epochs"])
        self.batch_size = int(self.nn_config["batch_size"])
        self.checkpoint_every = int(self.training_config["checkpoint_every"])

        self.num_conv=int(self.nn_config["num_conv"])
        self.node_type_dim=int(self.nn_config["node_type_dim"])
        self.node_token_dim=int(self.nn_config["node_token_dim"])
        self.conv_output_dim=int(self.nn_config["conv_output_dim"]) 
        self.include_token=int(self.nn_config["include_token"])
        self.batch_size=int(self.nn_config["batch_size"])
        self.learning_rate=float(self.nn_config["lr"])
    
    def init_resources(self):
        self.data_path = self.resource_config["data_path"]
        self.output_processed_data_path = self.resource_config["output_processed_data_path"]
        self.model_name = self.resource_config["model_name"]
        self.pretrained_model_url = self.resource_config["pretrained_model_url"]

        # Init vocab
        self.node_type_vocab_model_prefix = os.path.join(package_dir, "sentencepiece_vocab", self.resource_config["node_type_vocab_model_prefix"])
        self.node_token_vocab_model_prefix = os.path.join(package_dir, "sentencepiece_vocab", self.resource_config["node_token_vocab_model_prefix"])
        self.subtree_vocab_model_prefix = os.path.join(package_dir, "sentencepiece_vocab", self.resource_config["subtree_vocab_model_prefix"])

    def init_model_checkpoint(self):
        home = str(Path.home())
        cd = os.getcwd()
        model_checkpoint = path.join(home, ".infercode_data" ,"model_checkpoint", self.model_name)
        model_checkpoint_ckpt = path.join(model_checkpoint, "cnn_tree.ckpt.index")

        if not os.path.exists(model_checkpoint):
            pathlib.Path(model_checkpoint).mkdir(parents=True, exist_ok=True)

        """
        Comment out this part if training locally
        """
        if not os.path.exists(model_checkpoint_ckpt):
            pretrained_model_checkpoint_target = os.path.join(model_checkpoint, "universal_model_med.zip")
            download_url(self.pretrained_model_url, pretrained_model_checkpoint_target)
            with zipfile.ZipFile(pretrained_model_checkpoint_target, 'r') as zip_ref:
                zip_ref.extractall(model_checkpoint)

        self.model_checkpoint = model_checkpoint
        
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