import argparse
import random

import pickle

import tensorflow as tf
from utils.data.tree_loader import CodeClassificationData
from utils.utils import ThreadedIterator
# from utils.network.dense_ggnn_method_name_prediction import DenseGGNNModel
from utils.network.infercode_network import InferCodeModel
# import utils.network.treecaps_2 as network
import os
import sys
import re
import time

from bidict import bidict
import copy
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import evaluation
from scipy.spatial import distance
from datetime import datetime
from keras_radam.training import RAdamOptimizer
import logging
logging.basicConfig(filename='training.log',level=logging.DEBUG)


np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument('--worker', type=int,
                    help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int,
                    default=4, help='input batch size')
parser.add_argument('--train_batch_size', type=int,
                    default=5, help='train input batch size')
parser.add_argument('--test_batch_size', type=int,
                    default=5, help='test input batch size')
parser.add_argument('--val_batch_size', type=int,
                    default=5, help='val input batch size')
parser.add_argument('--node_type_dim', type=int, default=30,
                    help='node type dimension size')
parser.add_argument('--node_token_dim', type=int,
                    default=30, help='node token dimension size')
parser.add_argument('--hidden_layer_size', type=int,
                    default=100, help='size of hidden layer')
parser.add_argument('--num_hidden_layer', type=int,
                    default=1, help='number of hidden layer')
parser.add_argument('--n_steps', type=int, default=10,
                    help='propagation steps number of GGNN')
parser.add_argument('--n_edge_types', type=int, default=7,
                    help='number of edge types')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--cuda', default="0", type=str, help='enables cuda')
parser.add_argument('--verbal', type=bool, default=True,
                    help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--model_path', default="model",
                    help='path to save the model')
# parser.add_argument('--model_accuracy_path', default="model_accuracy/method_name.txt",
#                     help='path to save the the best accuracy of the model')
parser.add_argument('--n_hidden', type=int, default=50,
                    help='number of hidden layers')
parser.add_argument('--log_path', default="logs/",
                    help='log path for tensorboard')
parser.add_argument('--checkpoint_every', type=int,
                    default=100, help='check point to save model')
parser.add_argument('--validating', type=int,
                    default=1, help='validating or not')
parser.add_argument('--tree_size_threshold_upper', type=int,
                    default=5000, help='tree size threshold')
parser.add_argument('--tree_size_threshold_lower', type=int,
                    default=30, help='tree size threshold')                   
parser.add_argument('--sampling_size', type=int,
                    default=60, help='sampling size for each epoch')
parser.add_argument('--best_f1', type=float,
                    default=0.0, help='best f1 to save model')
parser.add_argument('--aggregation', type=int, default=1, choices=range(0, 4),
                    help='0 for max pooling, 1 for attention with sum pooling, 2 for attention with max pooling, 3 for attention with average pooling')
parser.add_argument('--distributed_function', type=int, default=0,
                    choices=range(0, 2), help='0 for softmax, 1 for sigmoid')
parser.add_argument('--train_path', default="OJ_pkl_train_test_val/train",
                    help='path of training data')
parser.add_argument('--val_path', default="OJ_pkl_train_test_val/train",
                    help='path of validation data')
parser.add_argument('--train_label_path', default="OJ_stmt_train_test_val/train",
                    help='path of training data')
parser.add_argument('--val_label_path', default="OJ_stmt_train_test_val/train",
                    help='path of validation data')
parser.add_argument('--node_type_vocabulary_path', default="vocab/type_vocab.csv",
                    help='the path to node type vocab')
parser.add_argument('--token_vocabulary_path', default="vocab/token_vocab.csv",
                    help='the path to node token vocab')
parser.add_argument('--subtree_vocabulary_path', default="subtree_features/OJ_stmt_features_train.csv",
                    help='the path to subtree vocab')
parser.add_argument('--task', type=int, default=1,
                    choices=range(0, 2), help='0 for training, 1 for testing')
parser.add_argument('--loss', type=int, default=0,
                    choices=range(0, 3), help='loss function, 0 for softmax, 1 for sampled softmax, 2 for nce')
parser.add_argument('--num_sampling', type=int, default=2, help="number of subtrees to be sampled for sampled softmax")
parser.add_argument('--num_files_threshold', type=int, default=20000)
parser.add_argument('--top_a', type=int, default=10)
parser.add_argument('--num_conv', type=int, default=1)
parser.add_argument('--output_size', type=int, default=30)
parser.add_argument('--model_name', default="stmt")
parser.add_argument('--dataset', default="OJ_new")
parser.add_argument('--include_token', type=int,
                    default=1, help='including token for initializing or not, 1 for including, 0 for excluding')

opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda

# def train_model():

#     train_dataset = MethodNamePredictionData(opt, opt.train_path, True, False, False)

    # train_batch_iterator = ThreadedIterator(train_dataset.make_minibatch_iterator(), max_queue_size=10)
    # for train_step, train_batch_data in enumerate(train_batch_iterator):
    #     print("-------------------------------------")
    #     print(train_batch_data)

def form_model_path(opt):
    model_traits = {}
  
    model_traits["node_type_dim"] = str(opt.node_type_dim)
    model_traits["node_token_dim"] = str(opt.node_token_dim)
    model_traits["output_size"] = str(opt.output_size)
    model_traits["num_conv"] = str(opt.num_conv)
    model_traits["include_token"] = str(opt.include_token)
    # model_traits["version"] = "direct-routing"
    
    
    model_path = []
    for k, v in model_traits.items():
        model_path.append(k + "_" + v)
    


    return opt.dataset + "_" + opt.model_name + "_" + "sampled_softmax" + "_" + "-".join(model_path)

def load_vocabs(opt):

    node_type_lookup = {}
    node_token_lookup = {}
    subtree_lookup = {}

    node_type_vocabulary_path = opt.node_type_vocabulary_path
    token_vocabulary_path = opt.token_vocabulary_path
    subtree_vocabulary_path = opt.subtree_vocabulary_path

    with open(node_type_vocabulary_path, "r") as f2:
        data = f2.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            node_type_lookup[splits[1]] = int(splits[0])

    with open(token_vocabulary_path, "r") as f3:
        data = f3.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            node_token_lookup[splits[1]] = int(splits[0])

    with open(subtree_vocabulary_path, "r") as f4:
        data = f4.readlines()
        for i, line in enumerate(data):
            splits = line.replace("\n", "").split(",")
            subtree_lookup[splits[0]] = i


    node_type_lookup = bidict(node_type_lookup)
    node_token_lookup = bidict(node_token_lookup)
    subtree_lookup = bidict(subtree_lookup)

    return node_type_lookup, node_token_lookup, subtree_lookup

def get_best_f1_score(opt):
    best_f1_score = 0.0
    
    try:
        os.mkdir("model_accuracy")
    except Exception as e:
        print(e)
    
    opt.model_accuracy_path = os.path.join("model_accuracy",form_model_path(opt) + ".txt")

    if os.path.exists(opt.model_accuracy_path):
        print("Model accuracy path exists : " + str(opt.model_accuracy_path))
        with open(opt.model_accuracy_path,"r") as f4:
            data = f4.readlines()
            for line in data:
                best_f1_score = float(line.replace("\n",""))
    else:
        print("Creating model accuracy path : " + str(opt.model_accuracy_path))
        with open(opt.model_accuracy_path,"w") as f5:
            f5.write("0.0")
    
    return best_f1_score


def get_accuracy(target, sample_id):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], sample_id.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - sample_id.shape[1]:
        sample_id = np.pad(
            sample_id,
            [(0,0),(0,max_seq - sample_id.shape[1])],
            'constant')

    return np.mean(np.equal(target, sample_id))


def main(opt):
    
    opt.model_path = os.path.join(opt.model_path, form_model_path(opt))
    checkfile = os.path.join(opt.model_path, 'cnn_tree.ckpt')
    ckpt = tf.train.get_checkpoint_state(opt.model_path)
    print("The model path : " + str(checkfile))
    print("Loss : " + str(opt.loss))
    if ckpt and ckpt.model_checkpoint_path:
        print("Continue training with old model : " + str(checkfile))

    print("Loading vocabs.........")
    node_type_lookup, node_token_lookup, subtree_lookup = load_vocabs(opt)

    opt.node_type_lookup = node_type_lookup
    opt.node_token_lookup = node_token_lookup
    opt.subtree_lookup = subtree_lookup

    if opt.task == 1:
        train_dataset = CodeClassificationData(opt, True, False, False)

    if opt.task == 0:
        val_opt = copy.deepcopy(opt)
        val_opt.node_token_lookup = node_token_lookup
        validation_dataset = CodeClassificationData(val_opt, False, False, True)

    print("Initializing tree caps model...........")
    corder = CorderModel(opt)
    print("Finished initializing corder model...........")


    loss_node = corder.loss
    optimizer = RAdamOptimizer(opt.lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        training_point = optimizer.minimize(loss_node)
    saver = tf.train.Saver(save_relative_paths=True, max_to_keep=5)
  
    init = tf.global_variables_initializer()

    # best_f1_score = get_best_f1_score(opt)
    # print("Best f1 score : " + str(best_f1_score))



    with tf.Session() as sess:
        sess.run(init)
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training with old model")
            print("Checkpoint path : " + str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i, var in enumerate(saver._var_list):
                print('Var {}: {}'.format(i, var))

        if opt.task == 1:
            for epoch in range(1,  opt.epochs + 1):
                train_batch_iterator = ThreadedIterator(train_dataset.make_minibatch_iterator(), max_queue_size=opt.worker)
                train_accs = []
                for train_step, train_batch_data in enumerate(train_batch_iterator):
                    print("--------------------------")

                    # print(train_batch_data["batch_subtrees_ids"])
                    logging.info(str(train_batch_data["batch_subtree_id"]))
                    _, err = sess.run(
                        [training_point, corder.loss],
                        feed_dict={
                            corder.placeholders["node_types"]: train_batch_data["batch_node_types"],
                            corder.placeholders["node_tokens"]:  train_batch_data["batch_node_tokens"],
                            corder.placeholders["children_indices"]:  train_batch_data["batch_children_indices"],
                            corder.placeholders["children_node_types"]: train_batch_data["batch_children_node_types"],
                            corder.placeholders["children_node_tokens"]: train_batch_data["batch_children_node_tokens"],
                            corder.placeholders["labels"]: train_batch_data["batch_subtree_id"],
                            corder.placeholders["dropout_rate"]: 0.3
                        }
                    )

                    logging.info("Training at epoch " + str(epoch) + " and step " + str(train_step) + " with loss "  + str(err))
                    print("Epoch:", epoch, "Step:", train_step, "Training loss:", err)
                    if train_step % opt.checkpoint_every == 0 and train_step > 0:
                        saver.save(sess, checkfile)                  
                        print('Checkpoint saved, epoch:' + str(epoch) + ', step: ' + str(train_step) + ', loss: ' + str(err) + '.')
              


        if opt.task == 0:
            validation_batch_iterator = ThreadedIterator(validation_dataset.make_minibatch_iterator(), max_queue_size=opt.worker)         
           
            for val_step, val_batch_data in enumerate(validation_batch_iterator):
                scores = sess.run(
                    [corder.code_vector],
                    feed_dict={
                        corder.placeholders["node_types"]: val_batch_data["batch_node_types"],
                        corder.placeholders["node_tokens"]:  val_batch_data["batch_node_tokens"],
                        corder.placeholders["children_indices"]:  val_batch_data["batch_children_indices"],
                        corder.placeholders["children_node_types"]: val_batch_data["batch_children_node_types"],
                        corder.placeholders["children_node_tokens"]: val_batch_data["batch_children_node_tokens"],
                        corder.placeholders["dropout_rate"]: 0.0
                    }
                )
                

                for i, vector in enumerate(scores[0]):
                    file_name = "analysis/rosetta_sampled_softmax_train.csv"
                    with open(file_name, "a") as f:
                        vector_score = []
                        for score in vector:
                            vector_score.append(str(score))
                        # print(val_batch_data["batch_file_path"])
                        line = str(val_batch_data["batch_file_path"][i]) + "," + " ".join(vector_score)
                        f.write(line)
                        f.write("\n")
                    



if __name__ == "__main__":
    main(opt)
   