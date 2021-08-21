import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from client.infercode_client import InferCodeClient
import os
import logging
logging.basicConfig(level=logging.INFO)
import math
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
# Change from -1 to 0 to enable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


def calculate_cosine_distance(a, b):
    cosine_distance = float(spatial.distance.cosine(a, b))
    return cosine_distance


def calculate_cosine_similarity(a, b):
    cosine_similarity = 1 - calculate_cosine_distance(a, b)
    return cosine_similarity

infercode = InferCodeClient(language="c")
infercode.init_from_config()

with open("../../datasets/OJ_raw_small/1/1001.c", "r") as f1:
    f1_data = str(f1.read())

with open("../../datasets/OJ_raw_small/1/1008.c", "r") as f2:
    f2_data = str(f2.read())

vectors = infercode.encode([f1_data, f2_data])

print(vectors)

print(cosine_similarity([vectors[0]], [vectors[1]]))

#0.99997956
#0.9999984
