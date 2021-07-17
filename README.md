<p align="center">
    <br>
    <img src="logo/twitter_header_photo_2.png" width="600"/>
    <br>
<p>
  
## Map Any Code Snippet into Vector Embedding with InferCode. 

This is a Tensorflow Implementation for "InferCode: Self-Supervised Learning of Code Representations by Predicting Subtrees" (ICSE'21). InferCode works based on the key idea of using an encoder to predict subtrees as a pretext task. Then the weights learned from the encoder can be used to transfer for other downstream tasks. This is to alleviate the need for the huge amount of labeled data to build decent code learning models in Software Engineering. With this concept, representation learning models for  source code can now learn from unlabeled data. 

## Datasets
- SrcML AST format of java-small data: https://ai4code.s3.ap-southeast-1.amazonaws.com/java-small-pkl.zip
- SrcML AST format of OJ data: https://ai4code.s3.ap-southeast-1.amazonaws.com/OJ_raw_pkl.zip
- Raw OJ data from Mou et al.: https://ai4code.s3.ap-southeast-1.amazonaws.com/OJ_raw.zip

## Quick run
If you want to have a quick run to see how to infer the code vectors from some code snippets, please follow these steps:
- ```git clone git@github.com:bdqnghi/infercode.git```
- Download the small version of the OJ data from Peking University from this link: https://ai4code.s3.ap-southeast-1.amazonaws.com/OJ_raw_small.zip , then unzip it.
```bash
cd infercode
wget https://ai4code.s3.ap-southeast-1.amazonaws.com/OJ_raw_small.zip
unzip OJ_raw_small
```
- Next, convert the all of the code snippets in OJ_raw_small to SrcML AST format, run: 

```python
python3 generate_srcml_pkl.py --input_path OJ_raw_small --output_path OJ_raw_pkl_small
```
You will see an ```OJ_raw_pkl_small``` folder that is generated after this command.

- Next, we need to pre-process the data, run:
```bash
cd processing_script
python3 preprocess_data.py --input_data_directory ../OJ_raw_pkl_small --output_path ../OJ_raw_pkl_small/OJ_raw_pkl_small.pkl  \
--node_type_vocabulary_path ../vocab/type_vocab.csv --token_vocabulary_path ../vocab/java-small/token_vocab.csv --training 0
```
You will see a pickle file in the path ```OJ_raw_pkl_small/OJ_raw_pkl_small.pkl``` that is generated after this command. This file is the data that has been pre-processed for the deep learning training/inferring purpose.

- Finally, we can infer the code vectors from the pre-processed data:
```bash
python3 infer.py --data_path OJ_raw_pkl_small/OJ_raw_pkl_small.pkl --subtree_vocabulary_path subtrees_vocab/java-small_subtrees_vocab.csv \
--node_type_vocabulary_path vocab/type_vocab.csv --token_vocabulary_path vocab/java-small/token_vocab.csv --training 0
```

Then you can see an ```embeddings.csv``` file that have the content similar to this:

```
../OJ_raw_pkl_small/1/518.pkl,0.5255707 0.30696836 0.77132815 -0.6314099 0.50227034 -0.041038744 -0.50494325 -0.19209197 -1.775357 -0.17768924 -0.71590745 -0.21211924 -0.41921812 -0.7690327 0.08125296 -0.50575626 -0.3210821 0.68810135 -1.2741802 -0.62127084 -0.6283189 -0.45203987 -0.48960194 -0.57419443 -0.1258675 0.53587574 -0.3871129 0.4171466 2.3695114 0.024016896 0.79501593 -0.29024157 -0.44861493 -0.61119825 -0.15325235 0.08081369 0.25650156 -0.6703408 -0.53233975 -0.2777813 -0.38442627 -0.02405865 -0.44204476 -0.49111685 -0.35549343 0.24859273 -0.75000215 0.4976057 0.69803643 -0.066390306
../OJ_raw_pkl_small/4/917.pkl,0.40941036 0.7452002 0.8647637 -0.8030474 0.6020626 -0.075376384 -0.5745088 -0.5786715 -1.8721092 -0.9546244 -0.5602549 0.25473255 -0.97351164 -0.86666787 -0.19205664 -0.45276073 -0.8698674 0.41794208 -1.4720353 -0.37846515 -0.75341624 -0.6399622 -0.48165962 -0.2992712 0.045876585 1.9478942 -0.7625236 0.41821206 2.3921921 0.28091958 1.1714528 -0.34662044 -0.60124314 -0.66371685 -0.30104768 0.08676574 0.23696409 -0.7645121 -0.3333376 -0.49727312 -0.4702112 0.6312706 -0.49725312 0.11638083 -0.11783122 0.28807533 -0.5108802 0.17315492 1.1042275 -0.110817224
../OJ_raw_pkl_small/1/1939.pkl,0.39321837 0.51704186 0.42872062 -0.6258228 0.87729675 -0.0502351 -0.67181677 0.27130735 -1.9256694 -0.61644876 -0.6358697 -0.053383004 -0.487053 -0.48355898 -0.0037307574 -0.56086963 -0.35996243 0.38160872 -1.1647105 0.17598744 -0.7358754 -0.64036226 -0.24140441 -0.25993958 0.1454117 1.7947239 -0.45594862 0.3041636 2.0785716 0.011919543 0.58531165 -0.15157986 -0.11728069 -0.49890566 -0.11557173 -0.09826957 0.1523883 -0.6942044 -0.4125205 -0.38668665 -0.3302969 0.067926385 -0.63356084 -0.55278647 -0.41471916 0.11744309 -0.8110009 0.38603508 0.8242188 -0.22820777
```
Each line is the a file name that contain the code snippet and its corresponding code vectors.

## End-to-End Training Process


### Convert raw source code into SrcML AST
First, we need to convert the raw dataset into the SrcML AST. We have packaged the tool into docker and write a python script to execute the docker command. To generate the SrcML AST, run:

```python
python3 generate_srcml_pkl.py --input_path java-small/training --output_path java-small-pkl/training
```

### Generate Subtrees
Next, we need to generate the subtrees as the pseudo-label for training. We have packaged the tool to generate the subtrees into a docker image. To generate the subtrees, run:

```python
python3 generate_subtrees.py --input_path java-small --output_path java-small-subtrees --node_types_path node_types.csv
```

Note that you need to install Docker for the command to work.
- node_types.csv: contains the node types to consider in the AST. 
- input_folder: is the path to the directory that contains the raw source code files, e.g., .cpp, .c, .java, .cs.
- output_folder: is the path to the directory that contains the subtrees.

Each file in the output folder contains the subtrees, each subtree is in this format:
- ```rootid-roottype-roottoken,nodeid-nodetype-nodetoken nodeid-nodetype-nodetoken nodeid-nodetype-nodetoken ..., depth_of_the_subtree```
- ```rootid-roottype-roottoken```: is the information of the root node of a subtree
- ```nodeid-nodetype-nodetoken```: is the information of a node in a subtree

The subtrees are sequentialized using the DFS algorithm.

Once we have all of the subtrees for all of the samples in our training data, we need to merge them into a set of subtree vocabulary, to do so, run:

```python
cd processing_script
python3 extract_all_subtrees --input ../java-small-subtrees --output ../subtrees_vocab/java-small-subtrees-vocab.txt
```

### Preprocess the data for training
```bash
cd processing_script
source preprocess_data.sh
```

### Training the model
To start training, run:
```bash
source train.sh
```

### Inferring code vector from pretrained model
We have included our pretrained model on the java-small dataset in the directory ``model/``. To test the model, run:

```bash
source infer.sh
```

## Notes
- For a fair comparison with InferCode in the future, please consider
  + Using the same ASTs structure, since different ASTs structures can affect the performance significantly. We use SrcML in InferCode
  + Using the similar settings of InferCode on the embedding size (e.g. node type embedding, node token embedding

## Citation
If you find this tutorial useful for your research, please consider citing our paper:

```bibtex
@inproceedings{bui2021infercode,
  title={InferCode: Self-Supervised Learning of Code Representations by Predicting Subtrees},
  author={Bui, Nghi DQ and Yu, Yijun and Jiang, Lingxiao},
  booktitle={2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE)},
  pages={1186--1197},
  year={2021},
  organization={IEEE}
}
```
