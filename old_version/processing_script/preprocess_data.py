import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
import random
import pickle
from utils.data.tree_processor import TreeProcessor
import argparse
# import utils.network.treecaps_2 as network
import os
import re
import time
from bidict import bidict
import copy
import numpy as np
from utils import evaluation

def parse_arguments(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_type_dim', type=int, default=30,
                        help='node type dimension size')
    parser.add_argument('--node_token_dim', type=int,
                        default=30, help='node token dimension size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--cuda', default="0", type=str, help='enables cuda')
    parser.add_argument('--verbal', type=bool, default=True,
                        help='print training info or not')
    parser.add_argument('--training', type=int, default=0,
                        help='process data for training or testing')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--log_path', default="logs/",
                        help='log path for tensorboard')
    parser.add_argument('--checkpoint_every', type=int,
                        default=100, help='check point to save model')
    parser.add_argument('--tree_size_threshold_upper', type=int,
                        default=5000, help='tree size threshold')
    parser.add_argument('--tree_size_threshold_lower', type=int,
                        default=30, help='tree size threshold')                   
    parser.add_argument('--input_data_directory', default="java-small/training",
                        help='path of data directory to pre-process')
    parser.add_argument('--output_path', default="java-small/training",
                        help='output path in form of pickle format')
    parser.add_argument('--subtree_directory', default="java-small-subtrees/training",
                        help='path of data directory to pre-process')
    parser.add_argument('--label_path', default="OJ_stmt_train_test_val/train",
                        help='path of label to predict')
    parser.add_argument('--node_type_vocabulary_path', default="../vocab/type_vocab.csv",
                        help='the path to node type vocab')
    parser.add_argument('--token_vocabulary_path', default="../vocab/java-small/token_vocab.csv",
                        help='the path to node token vocab')
    parser.add_argument('--subtree_vocabulary_path', default="../subtrees_vocab/java-small_subtrees_vocab.csv",
                        help='the path to subtree vocab')
    parser.add_argument('--model', default="java-small",
                        help='name of model')
    opt = parser.parse_args()

    return opt

def main(opt):
    tree_processor = TreeProcessor(opt)
    tree_processor.process_data()

if __name__ == "__main__":
    opt = parse_arguments()
    main(opt)