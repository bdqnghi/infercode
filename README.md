<p align="center">
    <br>
    <img src="logo/twitter_header_photo_2.png" width="600"/>
    <br>
<p>
  
## Map Any Code Snippet into Vector Embedding with InferCode. 

This is a Tensorflow Implementation for "InferCode: Self-Supervised Learning of Code Representations by Predicting Subtrees" (ICSE'21). InferCode works based on the key idea of using an encoder to predict subtrees as a pretext task. Then the weights learned from the encoder can be used to transfer for other downstream tasks. This is to alleviate the need for the huge amount of labeled data to build decent code learning models in Software Engineering. With this concept, representation learning models for  source code can now learn from unlabeled data. 

## Usage
```python
from client.infercode_client import InferCodeClient
infercode = InferCodeClient()
vectors = infercode.encode(["for (i = 0; i < n; i++)", "struct book{ int num; char s[27]; }shu[1000];"])
print(vectors)  
```
    
Then we have the output embeddings:
```bash
[[ 0.00455336  0.00277071  0.00299444 -0.00264732  0.00424443  0.02380365
0.00802475  0.01927063  0.00889819  0.01684897  0.03249155  0.01853252
0.00930241  0.02532686  0.00152953  0.0027509   0.00200306 -0.00042401
0.00093602  0.044968   -0.0041187   0.00760367  0.01713051  0.0051542
-0.00033204  0.01757674 -0.00852873  0.00510181  0.02680481  0.00579945
0.00298177  0.00650377  0.01903037  0.00188015  0.00644581  0.02502727
-0.00599149  0.00339381  0.01834774 -0.0012807  -0.00413265  0.01172356
0.01524384  0.00769007  0.01364587 -0.00340345  0.02757765  0.03651286
0.01334631  0.01464784]
[-0.00017088  0.01376707  0.01347563  0.00545072  0.01674811  0.01347677
0.01061796  0.02521674  0.01205592  0.03466582  0.01449588  0.02479498
-0.00011303  0.01174722  0.00444653  0.01382409 -0.00396148 -0.00195686
0.00527923  0.03169966 -0.00935379  0.01904526  0.02334653 -0.00742705
0.00405659  0.0158342  -0.00599484  0.01687686  0.03012032  0.01365279
0.01936428  0.00576922  0.01786506  0.00244599  0.00816536  0.03116215
-0.00721357  0.01265837  0.029279    0.00394636  0.00475944  0.0057507
0.02005564  0.00345545  0.01078242  0.00763404  0.01771503  0.02223164
0.01541999  0.03995579]]
```
    
## Notes
- Our old implementation can be found in [old_version](old_version/). 
- For a fair comparison with InferCode, please consider:
    + The code encoder part, we used Tree-based CNN as the encoder. There should be a better code encoder in the future given the fast progressing of AI research. 
    + The pretext task, inventing a new pretext task also affects on the quality of the code embeddings.
    
## Citation
If you find this work useful for your research, please consider citing our paper:

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
