import argparse
import random
import pickle
import tensorflow.compat.v1 as tf
from utils.data.data_loader.base_data_loader import BaseDataLoader
from utils.threaded_iterator import ThreadedIterator
import os
import sys
import re
import copy
import time
import argument_parser
import copy
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from datetime import datetime
from keras_radam.training import RAdamOptimizer
import logging
from utils.network.infercode_network import InferCodeModel
from utils import util_functions
logging.basicConfig(filename='training.log',level=logging.DEBUG)
np.set_printoptions(threshold=sys.maxsize)
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

def main(train_opt, test_opt):

    train_opt.model_path = os.path.join(train_opt.model_path, util_functions.form_tbcnn_model_path(train_opt))
    checkfile = os.path.join(train_opt.model_path, 'cnn_tree.ckpt')
    ckpt = tf.train.get_checkpoint_state(train_opt.model_path)
    print("The model path : " + str(checkfile))
    if ckpt and ckpt.model_checkpoint_path:
        print("-------Continue training with old model-------- : " + str(checkfile))


    tbcnn_model = InferCodeModel(train_opt)
    tbcnn_model.feed_forward()

    train_data_loader = BaseDataLoader(train_opt.batch_size, train_opt.tree_size_threshold_upper, train_opt.tree_size_threshold_lower, train_opt.train_tree_path,  train_opt.train_bucket_path,  True)
    # test_data_loader = BaseDataLoader(test_opt.batch_size, test_opt.tree_size_threshold_upper, test_opt.tree_size_threshold_lower, test_opt.test_path, False)

    optimizer = RAdamOptimizer(train_opt.lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        training_point = optimizer.minimize(tbcnn_model.loss)
    saver = tf.train.Saver(save_relative_paths=True, max_to_keep=5)  
    init = tf.global_variables_initializer()

    best_f1 = test_opt.best_f1
    with tf.Session() as sess:
        sess.run(init)
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training with old model")
            print("Checkpoint path : " + str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i, var in enumerate(saver._var_list):
                print('Var {}: {}'.format(i, var))

        
        for epoch in range(1,  train_opt.epochs + 1):
            train_batch_iterator = ThreadedIterator(train_data_loader.make_minibatch_iterator(), max_queue_size=train_opt.worker)
            for train_step, train_batch_data in enumerate(train_batch_iterator):
                print("***************")
                
                print(train_batch_data["batch_subtree_id"])
                _, err = sess.run(
                        [training_point, tbcnn_model.loss],
                        feed_dict={
                            tbcnn_model.placeholders["node_type"]: train_batch_data["batch_node_type_id"],
                            tbcnn_model.placeholders["node_token"]:  train_batch_data["batch_node_sub_tokens_id"],
                            tbcnn_model.placeholders["children_index"]:  train_batch_data["batch_children_index"],
                            tbcnn_model.placeholders["children_node_type"]: train_batch_data["batch_children_node_type_id"],
                            tbcnn_model.placeholders["children_node_token"]: train_batch_data["batch_children_node_sub_tokens_id"],
                            tbcnn_model.placeholders["subtree_labels"]: train_batch_data["batch_subtree_id"],
                            tbcnn_model.placeholders["dropout_rate"]: 0.3
                        }
                    )

                print("Epoch:", epoch, "Step:",train_step,"Loss:", err, "Best F1:", best_f1)
                # if train_step % train_opt.checkpoint_every == 0 and train_step > 0:
                               
                #     #Perform Validation
                #     print("Perform validation.....")
                #     correct_labels = []
                #     predictions = []
                #     test_batch_iterator = ThreadedIterator(test_data_loader.make_minibatch_iterator(), max_queue_size=test_opt.worker)
                #     for test_step, test_batch_data in enumerate(test_batch_iterator):
                #         print("***************")

                #         print(test_batch_data["batch_size"])
                #         scores = sess.run(
                #                 [tbcnn_model.softmax],
                #                 feed_dict={
                #                     tbcnn_model.placeholders["node_type"]: test_batch_data["batch_node_type_id"],
                #                     tbcnn_model.placeholders["node_token"]:  test_batch_data["batch_node_sub_tokens_id"],
                #                     tbcnn_model.placeholders["children_index"]:  test_batch_data["batch_children_index"],
                #                     tbcnn_model.placeholders["children_node_type"]: test_batch_data["batch_children_node_type_id"],
                #                     tbcnn_model.placeholders["children_node_token"]: test_batch_data["batch_children_node_sub_tokens_id"],
                #                     tbcnn_model.placeholders["labels"]: test_batch_data["batch_labels_one_hot"],
                #                     tbcnn_model.placeholders["dropout_rate"]: 0.0
                #                 }
                #             )
                #         batch_correct_labels = list(np.argmax(test_batch_data["batch_labels_one_hot"],axis=1))
                #         batch_predictions = list(np.argmax(scores[0],axis=1))
                    
                #         print(batch_correct_labels)
                #         print(batch_predictions)

                #         correct_labels.extend(np.argmax(test_batch_data["batch_labels_one_hot"],axis=1))
                #         predictions.extend(np.argmax(scores[0],axis=1))

                #     print(correct_labels)
                #     print(predictions)
                #     f1 = float(f1_score(correct_labels, predictions, average="micro"))
                #     print(classification_report(correct_labels, predictions))
                #     print('F1:', f1)
                #     print('Best F1:', best_f1)
                #     # print(confusion_matrix(correct_labels, predictions))

                #     if f1 > best_f1:
                #         best_f1 = f1
                #         saver.save(sess, checkfile)                  
                #         print('Checkpoint saved, epoch:' + str(epoch) + ', step: ' + str(train_step) + ', loss: ' + str(err) + '.')



if __name__ == "__main__":
    train_opt = argument_parser.parse_arguments()
    
    test_opt = copy.deepcopy(train_opt)
    # test_opt.data_path = "OJ_rs/OJ_rs-buckets-test.pkl"

    os.environ['CUDA_VISIBLE_DEVICES'] = train_opt.cuda

    main(train_opt, test_opt)
   