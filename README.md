## Tensorflow Implementation for "InferCode: Self-Supervised Learning of Code Representations by Predicting Subtrees" (ICSE'21)

InferCode works based on the key idea of using an encoder to predict subtrees as a pretext task. Then the weights learned from the encoder can be used to transfer for other downstream tasks. This is to alleviate the need for the huge amount of labeled data to build decent code learning models in Software Engineering. With this concept,  representation models for code can now learn from unlabeled data. 

## Process
- Convert raw source code into SrcML AST as the protobuf format.
- Generate subtrees: First, need to generate pseudo-labels for self-supervised learning. In this case, the pseudo-lables are the subtrees.
- Train the model: Once the subtrees are generated (the directory subtree_features), we can start to train the model.
- Infer vector from raw code: Once the encoder is trained, we can use it to generate vector for any source code snippet. Unfortunately, our tool could not receive raw source code directly, the tool can only receive the AST. It is because we need to rely on an external tool to generate the AST representation of the code. So we need to convert the code into the AST first.


### Convert raw source code into SrcML AST

```python
python3 generate_srcml_pkl.py --input_path java-small/training --output_path java-small-pkl/training
```

### Generate Subtrees
We have packaged the tool to generate the subtrees into a docker image. To generate the subtrees, run:

```python
python3 generate_subtrees.py --input_path java-small --output_path java-small-subtrees --node_types-path node_types.csv"
```

Note that you need to install Docker for the command to work.
- node_types.csv: contains the node types to consider in the AST. 
- input_folder: is the path to the directory that contains the raw source code files, e.g., .cpp, .c, .java, .cs.
- output_folder: is the path to the directory that contains the subtrees.

Each file in the output folder contains the subtrees, each subtree is in this format:
- rootid-roottype-roottoken,nodeid-nodetype-nodetoken nodeid-nodetype-nodetoken nodeid-nodetype-nodetoken ..., depth_of_the_subtree
- rootid-roottype-roottoken: is the information of the root node of a subtree
- nodeid-nodetype-nodetoken: is the information of a node in a subtree

The subtrees are sequentialized using the DFS algorithm.


## Running the model

1. To train the model:
    - ```source train.sh```
    
2. To test the model:
    - ```source infer.sh```
  
The script ```infer.sh``` will generate a file ``embeddings.csv`` which contains the embeddings of the code snippets, for example:

```
../java-small-test/training/cassandra/PrimaryKeyRestrictionSetTest_testboundsAsClusteringWithSingleEqAndSliceRestrictions.pkl,1.1499425 -0.19412728 0.025818767 -0.2866059 0.19273856 -0.06809299 1.1991358 0.40147448 -0.97792214 -0.68117386 -0.0483394 -0.27027488 0.31322715 0.27028129 -0.5513973 0.28848505 -0.24859701 0.034147665 1.804145 2.4824371 -0.5267946 -0.23878224 -0.40670702 -0.7706362 -0.09361468 1.2538036 0.5394761 -0.1507038 -0.3530482 -0.30349588 0.53271616 -0.36247018 1.4977133 1.4030226 -0.08373651 0.4650672 0.28952408 0.047818244 -0.39104933 -0.4957824 0.31893227 0.28905505 -0.11106472 1.3183858 -0.8878206 -0.3408521 -0.77557135 -0.77547204 -0.39631933 -0.08504311
../java-small-test/training/cassandra/IndexHelperTest_testIndexHelper.pkl,-0.67515516 -0.31055018 0.5445187 -0.38619745 -0.07080796 -0.38975635 3.4238505 1.0262927 -0.17105964 0.26446548 1.3404844 -0.6687252 1.5965147 -0.8458063 -0.3719226 -0.100242674 -0.60833764 0.7476521 1.2193403 -0.43699533 -0.6390345 -0.46230033 -0.5707703 -1.2751414 -0.035711195 -0.108094975 0.8059963 0.2387354 0.80078334 -0.027112097 0.21629927 -0.8486311 -0.43314686 2.4760997 0.06343947 0.043595485 -0.57052195 0.5428629 -0.2870741 -0.71483994 0.31383833 0.17307009 -0.6394538 0.09850803 -0.58634245 0.2265188 -0.79103243 -1.0044403 -0.5319375 0.41203216
../java-small-test/training/cassandra/OutboundTcpConnection_connect.pkl,-0.0062400224 0.057965927 -0.1212681 0.6923716 -0.027303757 0.15986258 -0.30424973 -0.040359445 -1.7294773 0.24670187 0.108225934 0.2529705 -0.039934266 -0.63573647 -0.017600775 0.43676466 -0.7703457 0.1892839 0.46893278 0.28055465 -0.0872712 -0.26793283 -0.26982522 0.022070276 1.7358426 -0.4967658 0.5286317 0.5311712 0.6177526 -0.6457953 -0.10148286 0.14781196 -0.6264381 -0.6947708 1.0489111 -0.11257442 -0.34575447 -0.14776468 0.6568464 -0.5123066 0.06919636 0.8075724 -0.12521954 -0.2027752 -0.67887807 -0.34929115 0.32236084 -0.0649846 -0.016871044 0.09877084
.....
```
From the above lines, the code snippet ``../java-small-test/training/cassandra/PrimaryKeyRestrictionSetTest_testboundsAsClusteringWithSingleEqAndSliceRestrictions.pkl`` has the embedding of 50 dimesions ``1.1499425 -0.19412728 0.025818767 -0.2866059 0.19273856 -0.06809299 1.1991358 0.40147448 -0.97792214 -0.68117386 -0.0483394 -0.27027488 0.31322715 0.27028129 -0.5513973 0.28848505 -0.24859701 0.034147665 1.804145 2.4824371 -0.5267946 -0.23878224 -0.40670702 -0.7706362 -0.09361468 1.2538036 0.5394761 -0.1507038 -0.3530482 -0.30349588 0.53271616 -0.36247018 1.4977133 1.4030226 -0.08373651 0.4650672 0.28952408 0.047818244 -0.39104933 -0.4957824 0.31893227 0.28905505 -0.11106472 1.3183858 -0.8878206 -0.3408521 -0.77557135 -0.77547204 -0.39631933 -0.08504311``



## Notes
- For a fair comparison with InferCode in the future, please consider
  + Using the same ASTs structure, since different ASTs structures can effect the performance a lot. We use SrcML in InferCode
  + Using the similar settings of InferCode on the embedding size
