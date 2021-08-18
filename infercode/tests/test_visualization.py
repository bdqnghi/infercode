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
logging.basicConfig(level=logging.INFO)
# import configparser 
# config = configparser.ConfigParser()
# config.read("../configs/OJ_raw_small.ini")
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
infercode = InferCodeClient(language="c")
infercode.init_from_config()
vocabulary = []
wv = []

colors = ["green", "black", "brown", "purple", "gold", "deeppink"]
labels = [1,2,3,4,5,6]
label_color_dict = dict(zip(labels, colors))


for subdir , dirs, files in os.walk("../../datasets/OJ_raw_small"): 
    for file in tqdm(files):
        file_path = os.path.join(subdir, file)
        file_path_splits = file_path.split("/")
        label = file_path_splits[len(file_path_splits)-2]
        vocabulary.append(label)
        snippet = str(open(file_path, "r"))
        vector = infercode.encode([snippet])
        wv.append(vector[0])

print(vocabulary)
print(wv)

tsne = TSNE(n_components=2, random_state=0)

np.set_printoptions(suppress=True)
Y = tsne.fit_transform(wv)


# plt.scatter(Y[:,0],Y[:,1], color='r')
print("Annotating....")
for i, point in enumerate(Y[:,0]):
    x = point
    y = Y[:,1][i]
    label = vocabulary[i]
    color = label_color_dict[int(label)]
    plt.scatter(x, y, color =color)


plt.title("Visualization")
# plt.show()

output = "test.png"

# figure = plt.gcf()

# figure.set_size_inches(8, 6)
plt.savefig(output)

