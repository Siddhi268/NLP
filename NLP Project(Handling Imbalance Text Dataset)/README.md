# Twitter Sentiment Analysis
A project is about handling text imbalanced classification with data augmentation technique.

## Goal
To handle imbalance text data using data augmentation techniques. 

## Project Introduction
It is often that the data we retrieve have imbalanced label and we are asked to make classification.And aiming only for high accuracy for the imbalanced dataset can be counter productive.As we can reach an accuracy of 95% by simply predicting majority class every time, but this provides a useless results as class with fewer data points are treated as noise and are often ignored for your intended use case. Instead, a properly calibrated method may achieve a lower accuracy but would have a substantially higher true positive rate (or recall), which is really the metric you should have been optimizing for.Therefore, the accuracy metric is not as relevant when evaluating the performance of a model trained on imbalanced data.The approaches for the project are :

   1. Data Cleaning (removing special characters,numbers,hashtags,etc).
   2. Randomly split the dataset into train and test data.
   3. Perform data augmentation on training data.
   4. Data Pre-Processing of training data(which include tokenization,text to sequence,padding). 
   5. Data Pre-Processing of testing data(which include text to sequence,padding).
   6. Apply Artificial Neural Network(like RNN,LSTM,GRU) .
   7. Compare the difference between the predictions and choose the best artificial neural network.

## Data Augmentation
Data Augmentation is the practice of synthesizing new data from data at hand. This could be applied to any form of data from numbers to images. Usually, the augmented data is similar to the data that is already available.

In the project we have used to two technique to handle the the imbalance data:

1.NLPAUG:

NLPAug is a python library for textual augmentation in machine learning experiments. The goal is to improve deep learning model performance by generating textual data. NLPAug is a tool that assists you in enhancing NLP for machine learning applications.

NLPAug provides three different types of augmentation:

1.Character level augmentation

2.Word level augmentation

3.Flow/ Sentence level augmentation

From the above three different type we have implemented word level augmentation with Synonym Augmenter.Synonym Augmenter applies semantic meaning based on textual input.

Implemented the nlpaug with different Artificial Neural Network and best results are obtained by Bidirectional LSTM.


![result1](https://user-images.githubusercontent.com/73767113/145040180-11ad017a-7722-4be4-85b1-fa5707f9d701.jpg)


#### Link to Jupyter Notebook in Repository
[Notebook](https://github.com/Siddhi268/NLP/blob/main/NLP%20Project(Handling%20Imbalance%20Text%20Dataset)/Data_augmentation_using_nlpaug%20.ipynb)


2.WORD EMBEDDING:

Word embedding is a language modelling technique to represent the words or phrases as vectors. The words are grouped together to get similar representation for words with similar meaning. The word embedding learns the relationship between the words to construct the representation. This is achieved by the various methods like co-occurrence matrix, probabilistic modelling, neural networks.

In word embedding ,i have implemented pre-trained word embedding i.e.Glove.And the results of neural network is as below:


![result2](https://user-images.githubusercontent.com/73767113/145040309-89ecc09e-473c-4f38-8ac7-5733d75e8fdb.jpg)


#### Link to Jupyter Notebook in Repository
[Notebook](https://github.com/Siddhi268/NLP/blob/main/NLP%20Project(Handling%20Imbalance%20Text%20Dataset)/Data_augmentation_using_word_embedding.ipynb)
