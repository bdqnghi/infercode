## Tensorflow Implementation for "InferCode: Self-Supervised Learning of Code Representations by Predicting Subtrees" (ICSE'21)

InferCode works based on the key idea of using an encoder to predict subtrees as a pretext task. Then the weights learned from the encoder can be used to transfer for other downstream tasks. This is to alleviate the need for the huge amount of labeled data to build decent code learning models in Software Engineering. With this concept,  representation models for code can now learn from unlabeled data. 

## Generate Subtrees
We have packaged the tool to generate the subtrees into a docker image
```docker run --rm -v $(pwd):/data -w /data --entrypoint /usr/local/bin/subtree -it yijun/fast input_folder output_folder node_types.csv```
- input_folder: is the path to the directory that contains the raw source code files, e.g., .cpp, .c, .java, .cs
- output_folder: is the path to the directory that contains the subtrees, each subtree is in this format:
rootid-roottype-roottoken,nodeid-nodetype-nodetoken nodeid-nodetype-nodetoken nodeid-nodetype-nodetoken ..., depth_of_the_subtree

rootid-roottype-roottoken: is the information of the root node of a subtree

nodeid-nodetype-nodetoke: is the information of a node in a subtree

The subtrees are sequentialized using the DFS algorith,



## Data Preparation

1. Install the required dependencies ```pip install -r requirements.txt```.

2. Download and extract the dataset and the pretrained models;

    -```cd script```

    -```python3 download_data.py```


3. Preprocess the data
    -```cd script```
    
    -```source process_data.sh```

This step will process the AST trees, which comprises of 2 steps. First, it will convert the pycparser format into our simple tree format in the form of Python dictionary. Second, it will bucket the trees with similar sizes into the same bucket.



## Running the model

1. To train the model:
    - ```source training_script.sh```
    
2. To test the model:
    - ```source esting_script.sh```
  

## Notes
- For a fair comparison with InferCode in the future, please consider
  + Using the same ASTs structure, since different ASTs structures can effect the performance a lot. We use SrcML in InferCode
  + Using the similar settings of InferCode on the embedding size
