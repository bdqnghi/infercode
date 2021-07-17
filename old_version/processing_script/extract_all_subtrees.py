import numpy as np
import os
from os import listdir
from os.path import isfile, join
import collections
import re
from tqdm import trange
from tqdm import *
import random
#import pickle
import pickle
import hashlib
import csv
import collections
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--worker", default=6, type=int, help="Num worker")
parser.add_argument("--input", default="../OJ_raw_pkl_subtrees", type=str, help="Input path")
parser.add_argument("--output", default="../subtrees_vocab/OJ_subtrees_vocab.csv", type=str, help="Input path")


def load_features(features_file_path):

    file_subtrees_dict = {}
    file_subtrees = []
    with open(features_file_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter= ",")
        for row in reader:
            # print("-------------")
            root_node = row[0]
            subtree_nodes = row[1]
            depth = row[2]

            root_id = root_node.split("-")[0]
            root_type = root_node.split("-")[1]
           
            subtree_nodes = subtree_nodes.split(" ")

            # print(root)
            # print(subtrees)
            single_subtree = [root_type]
            for subtree_node in subtree_nodes:
                # print(subtree)
                if len(subtree_node.split("-")) == 3:
                    subtree_node_id = subtree_node.split("-")[0]
                    subtree_node_type = subtree_node.split("-")[1]
                    single_subtree.append(subtree_node_type)

       
            str_features = "_".join(single_subtree)
            # hash_object = hashlib.md5(str_features.encode()).hexdigest()

            # file_subtrees_dict[hash_object] = {}
            # file_subtrees_dict[hash_object]["str_features"] = str_features
            # file_subtrees_dict[hash_object]["depth"] = depth
            file_subtrees.append(str_features)
            # print(whole_subtree_node_types)

    return file_subtrees


def main(args):

    input = args.input
    output = args.output

    # all_hash_features = {}
    all_subtrees = []
    for subdir , dirs, files in os.walk(input): 
        for file in tqdm(files):
            file_path = os.path.join(subdir, file)
            file_subtrees = load_features(file_path)
            # all_hash_features.update(file_subtrees_dict)
            all_subtrees.extend(file_subtrees)

    counter = collections.Counter(all_subtrees)

    with open(output, "w") as f:
        for k, v in counter.items():
            if counter[k] > 1:
                hash_object = hashlib.md5(k.encode()).hexdigest()
                line = str(hash_object) + "," + str(k) + "," + str(counter[k])
                f.write(line)
                f.write("\n")


if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)