import argparse
import random

import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils.data.tree_loader import TreeLoader
from utils.threaded_iterator import ThreadedIterator
# from utils.network.dense_ggnn_method_name_prediction import DenseGGNNModel
from utils.network.infercode_network import InferCodeModel
# import utils.network.treecaps_2 as network
import os
import sys
import re
import time
import argument_parser 
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
        train_dataset = TreeLoader(opt, True, False, False)

    if opt.task == 0:
        val_opt = copy.deepcopy(opt)
        val_opt.node_token_lookup = node_token_lookup
        validation_dataset = TreeLoader(val_opt, False, False, True)

    print("Initializing tree caps model...........")
    infercode = InferCodeModel(opt)
    print("Finished initializing corder model...........")


    loss_node = infercode.loss
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
                    print(train_batch_data["batch_subtree_id"])
                    # print(train_batch_data["batch_subtrees_ids"])
                    logging.info(str(train_batch_data["batch_subtree_id"]))
                    _, err = sess.run(
                        [training_point, infercode.loss],
                        feed_dict={
                            infercode.placeholders["node_types"]: train_batch_data["batch_node_types"],
                            infercode.placeholders["node_tokens"]:  train_batch_data["batch_node_tokens"],
                            infercode.placeholders["children_indices"]:  train_batch_data["batch_children_indices"],
                            infercode.placeholders["children_node_types"]: train_batch_data["batch_children_node_types"],
                            infercode.placeholders["children_node_tokens"]: train_batch_data["batch_children_node_tokens"],
                            infercode.placeholders["labels"]: train_batch_data["batch_subtree_id"],
                            infercode.placeholders["dropout_rate"]: 0.3
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
                    file_name = "analysis/" + opt.dataset + "_" + opt.model_name + "_sampled_softmax_train.csv"
                    with open(file_name, "a") as f:
                        vector_score = []
                        for score in vector:
                            vector_score.append(str(score))
                        line = str(val_batch_data["batch_labels"][i]) + "," + " ".join(vector_score)
                        f.write(line)
                        f.write("\n")
                    



if __name__ == "__main__":
    opt = argument_parser.parse_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda
    
    main(opt)
