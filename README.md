## Tensorflow Implementation for InferCode, a novel self-supervised learning method on source code representations trained on unlabeled data (ICSE'21)

InferCode works based on the key idea of using an encoder to predict subtrees as a pretext task. Then the weights learned from the encoder can be used to transfer for other downstream tasks. This is to alleviate the need for the huge amount of labeled data to build decent code learning models in Software Engineering. With this concept,  representation models for code can now learn from unlabeled data. 

We provide the source code implementation of InferCode and parts of our dataset. The full dataset will be released in the camera-ready version.

### Why self-supervised learning?

In program representation learning, we need lots of labeled data to build the models. Most of the source code models at the moment are supervised, which requires label to train the model. Human annotators usually label the data manually, which is a time-consuming and costly process. Thus the label generation aspect is a big bottleneck in the current supervised learning paradigm. 

### What is self-supervised learning?
Self-supervised learning is a method that poses the following question to formulate an unsupervised learning problem as a supervised one. We want to have an answer to this question:
"Can we design the task in such a way that we can generate virtually unlimited labels from our existing samples and use that to learn the representations?"

In self-supervised learning, we substitute the block of human annotation with creatively leveraging some data property to construct a pseudo-supervised task. For example, in visual learning, instead of marking images as cat/dog, we might instead rotate them by 0/90/180/270 degrees and train a model to predict spin. We can produce virtually limitless training data from millions of photos that we have free online. Noted that we do not care about the final performance of this rotation predicting task but we can about the weights of the neural network that has been learned. There are many ways to introduce these kinds of human-invented pretext tasks for self-supervised learning in computer vision. 

InferCode is inspired by this idea, we invent a new task to predict the subtrees with the intuition that similar code fragments should contain similar subtrees. After the training, we have a pre-trained InferCode model with Tree-based CNN as the source code **encoder**. The pre-trained InferCode can use 2 two main use cases:
- The pre-trained InferCode can receive any AST representation of code snippets, then produce a vector representation of it for a downstream task.
- The weights from the pre-trained InferCode can be used in a fine-tuning process for other supervised learning tasks, which speed up the training time and can handle a small dataset. **We adapted this use case into 2 tasks: code classification and method name prediction. A good analogy for this use case in visual learning is to train Convolutional Neural Network (CNN) for medical image processing**. We don't usually have a large dataset of medical images for training, as such, computer vision researchers usually use pre-trained CNN models, such as GoogleNet, ImageNet for the fine-tuning process to solve this challenge. 


## Usage

To train the model:
- Run ```python3 main_infercode.py``` 

## Some References
Here are some of the inspirational presentations about self-supervised learning:
- [Yann Lecun's talk on self-supervised learning: could machine learn like humans?](https://www.youtube.com/watch?v=7I0Qt7GALVk&t=2639s)
- [DeepMind tutorial on Self-supervised learning at ICML 2020](https://drive.google.com/file/d/1Ee2_EBgJQY5rMEiZJaRxxs6Il7m3EA-o/view)
