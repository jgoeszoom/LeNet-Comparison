# Increasingly Higher-level CNN Framework Overhead Cost Using LeNet As Case Study # 

## Background ## 
  Machine Learning is the process of teaching various programs to identify various objects, patterns, or even to predict future outcomes. This process can be achieved with various APIs in python, C, or C++. These APIs have also grown to the point to where APIs are often developed with other APIs used as backends. But are these newer APIs always more efficient for machine learning than the APIs they use as backends? Does Keras, which uses Tensorflow as a backend, which uses cuDNN as a backend, offer the best performance for machine learning? 

## Project Overview ##
  From the lowest-level (cuDNN) to increasingly higher-level frameworks such as TensorFlow and eventually Keras, we wanted to observe the possible increase in overhead as well as compare trade-offs between the aforementioned progamming APIs. The main comparison between the three APIs will be of time to infer an input image. However, a look at difficulty in implementation as well as the tradeoff in control offered will also be discussed.

## Authors ## 
* Jack Gu
* Joseph Gozum
* Justin Lam

## Built With ##
* **C++** - Programming Language
* **Python** - Programming Language
* **cuDNN** - Deep Neural Network API developed by Nvidia
* **Tensorflow** - Framework for numerical computation using data flow graphs
* **Keras** - High-level framework used to create Convolution Neural Networks

## Acknowledgements ##
* [Daniel Wong](danielwong.org) - Teaching us about GPU programming and believing that we could even do this project
* [Peter Goldsborough](http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/) - Article helped get familiar with cuDNN API

# Bugs #
* Usage of cuBLAS in cuDNN code is not correct.
