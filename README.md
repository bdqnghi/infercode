<p align="center">
    <br>
    <img src="logo/twitter_header_photo_2.png" width="600"/>
    <br>
<p>
  
## Map Any Code Snippet into Vector Embedding with InferCode. 

This is a Tensorflow Implementation for "InferCode: Self-Supervised Learning of Code Representations by Predicting Subtrees" (ICSE'21). InferCode works based on the key idea of using an encoder to predict subtrees as a pretext task. Then the weights learned from the encoder can be used to transfer for other downstream tasks. This is to alleviate the need for the huge amount of labeled data to build decent code learning models in Software Engineering. With this concept, representation learning models for  source code can now learn from unlabeled data. 

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
