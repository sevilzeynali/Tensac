
# Tensac (Tensorflow Scientific Articles Classification)

This is a program in Python for classifing the scientific articles with Tensorflow and the neural networks. It uses Python [Tensorflow](https://www.tensorflow.org/) library.

## Installing and requirements

You need Python >= 2.6 or >= 3.3

You should install tensorflow 1.4.0

This program uses only the CPU and not GPU

## Preparing your data sets
for using this program you need to have data train as a vector and the labels of this data train for training and creating your model.
You need also a test data as like of your data train it should be vectorize you need also the labels of this data test for testing the performance of your model.

You can make a file with one article abstract per line. You can vectorize this corpora with [fastText](https://fasttext.cc/).
You should also have your labels or gategories for these abstracts in another text file. each line of this file refers to each line of the abstract file. so that the first line of your label file is the category of your first abstract in abstract file.
Your testing data should have the same format. You creat a text file of data test and a text file of labels.



## How to use

You need only put the paths in : 
```
TRAIN_DATA_PATH=''
TRAIN_LABEL_PATH=''
TEST_DATA_PATH=''
TEST_LABEL_PATH=''
PREDICTION_PATH=''
LOG_PATH = ''
MODEL_PATH=''
```
Then you can execute this program in Python.
To see your graph, your histograms and your scalars in Tensorboard you can do :
```
--logdir=run1:[LOG_PATH] --port 6006
```
With your navigator in localhost you can visualise learning.
```
http://localhost:6006/
