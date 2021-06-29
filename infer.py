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
    
    return opt.model + "_" + "sampled_softmax" + "_" + "-".join(model_path)


def main(opt):
    
    opt.model_path = os.path.join(opt.model_path, form_model_path(opt))
    checkfile = os.path.join(opt.model_path, 'cnn_tree.ckpt')
    ckpt = tf.train.get_checkpoint_state(opt.model_path)
    print("The model path : " + str(checkfile))
    print("Loss : " + str(opt.loss))
    if ckpt and ckpt.model_checkpoint_path:
        print("Continue training with old model : " + str(checkfile))
  
    validation_dataset = TreeLoader(opt, training=False)
  
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


        validation_batch_iterator = ThreadedIterator(validation_dataset.make_minibatch_iterator(), max_queue_size=opt.worker)         
       

        for val_step, val_batch_data in enumerate(validation_batch_iterator):
            
            scores = sess.run(
                [infercode.code_vector],
                feed_dict={
                    infercode.placeholders["node_types"]: val_batch_data["batch_node_types"],
                    infercode.placeholders["node_tokens"]:  val_batch_data["batch_node_tokens"],
                    infercode.placeholders["children_indices"]:  val_batch_data["batch_children_indices"],
                    infercode.placeholders["children_node_types"]: val_batch_data["batch_children_node_types"],
                    infercode.placeholders["children_node_tokens"]: val_batch_data["batch_children_node_tokens"],
                    infercode.placeholders["dropout_rate"]: 0.0
                }
            )
            

            
            with open(opt.output_embedding_path, "w") as f:
                for i, vector in enumerate(scores[0]):
                    vector_score = []
                    for score in vector:
                        vector_score.append(str(score))
                    line = str(val_batch_data["batch_file_path"][i]) + "," + " ".join(vector_score)
                    f.write(line)
                    f.write("\n")
                    



if __name__ == "__main__":
    opt = argument_parser.parse_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda

    main(opt)
