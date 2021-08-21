import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from client.infercode_client import InferCodeClient
import os
import numpy as np
from sklearn.manifold import TSNE
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
from sklearn.cluster import DBSCAN
np.set_printoptions(threshold=sys.maxsize)
# import configparser 
# config = configparser.ConfigParser()
# config.read("../configs/OJ_raw_small.ini")
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
infercode = InferCodeClient(language="c")
infercode.init_from_config()
# vocabulary = []
# wv = []

# wv_1 = []
# wv_2 = []

def load_wv(path):
    wv = []
    for subdir , dirs, files in os.walk(path): 
        for file in tqdm(files):
            file_path = os.path.join(subdir, file)
            file_path_splits = file_path.split("/")
            # label = file_path_splits[len(file_path_splits)-2]
            # vocabulary.append(label)
            with open(file_path, "rb") as f:
                data = f.read()
            vector = infercode.encode([data])
            # print(vector)
            wv.append(vector[0])

    return wv


wv_1 = load_wv("../../datasets/OJ_raw_small/1")
wv_2 = load_wv("../../datasets/OJ_raw_small/5")


for v1 in wv_1:
    for v2 in wv_2:
        dist = np.linalg.norm(v1-v2)
        with open("dist.csv", "a") as f:
            f.write(str(dist))
            f.write("\n")
# np_arr = np.fromfile('OJ_raw_small.dat', dtype=float)
# print(np_arr.shape)
# db = DBSCAN(eps=0.01, min_samples=10).fit(np_arr)
# labels = db.labels_
# print(labels)

