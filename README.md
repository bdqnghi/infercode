## Tensorflow Implementation for "InferCode: Self-Supervised Learning of Code Representations by Predicting Subtrees" (ICSE'21)

InferCode works based on the key idea of using an encoder to predict subtrees as a pretext task. Then the weights learned from the encoder can be used to transfer for other downstream tasks. This is to alleviate the need for the huge amount of labeled data to build decent code learning models in Software Engineering. With this concept,  representation models for code can now learn from unlabeled data. 

## Process
- Convert raw source code into SrcML AST as the protobuf format.
- Generate subtrees: First, need to generate pseudo-labels for self-supervised learning. In this case, the pseudo-lables are the subtrees.
- Train the model: Once the subtrees are generated (the directory subtree_features), we can start to train the model.
- Infer vector from raw code: Once the encoder is trained, we can use it to generate vector for any source code snippet. Unfortunately, our tool could not receive raw source code directly, the tool can only receive the AST. It is because we need to rely on an external tool to generate the AST representation of the code. So we need to convert the code into the AST first.


### Convert raw source code into SrcML AST:

```python
python3 generate_srcml_pkl.py
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
  

## Notes
- For a fair comparison with InferCode in the future, please consider
  + Using the same ASTs structure, since different ASTs structures can effect the performance a lot. We use SrcML in InferCode
  + Using the similar settings of InferCode on the embedding size
