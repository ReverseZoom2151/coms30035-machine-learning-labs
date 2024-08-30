# COMS30035 Machine Learning 

This unit covers various Machine Learning (ML) exercises and implementations, including:

1. **Principal Component Analysis (PCA) on the MNIST dataset:** In this exercise, we apply PCA to the MNIST dataset of handwritten digits to reduce the dimensionality of the data while preserving as much variance as possible. We then reconstruct the data using the first K components and visualize the results.
2. **Independent Component Analysis (ICA) on mixed images:** In this exercise, we use ICA to separate three individual sources from a mixed signal, represented by three images. We compare the results of ICA to traditional PCA and show that ICA is able to separate the sources more effectively.
3. **Decision Trees and Ensembles on classification datasets:** In this exercise, we train decision tree classifiers on various classification datasets and evaluate their performance. We then implement bagging and random forests to improve the accuracy and diversity of the models.
4. **Hidden Markov Models (HMM) for part-of-speech tagging:** In this exercise, we use HMMs to perform part-of-speech tagging on the Brown corpus dataset. We implement both the emission and transition models for the HMM and use the Viterbi algorithm to decode the tags for a given sequence of words.
5. **Gaussian Mixture Models (GMM) for image segmentation:** In this exercise, we use GMMs to perform image segmentation on a synthetic dataset. We implement the expectation-maximization algorithm to estimate the parameters of the GMM and use the resulting model to segment the images.
6. **Neural Networks for digit recognition:** In this exercise, we train a neural network to recognize handwritten digits using the MNIST dataset. We implement a simple feedforward neural network with one hidden layer and use backpropagation to optimize the weights.
7. **Bayesian Inference for linear regression:** In this exercise, we use Bayesian inference to perform linear regression on a synthetic dataset. We define prior distributions for the model parameters and use Markov chain Monte Carlo (MCMC) to sample from the posterior distribution.

## Requirements
To run this project, you will need the following dependencies:

```
Python 3.x
NumPy
Matplotlib
scikit-learn
NLTK
PyMC3
TensorFlow
```

## Results
The results of each exercise are presented in the corresponding notebook. For some exercises, such as PCA and ICA, visualizations are provided to help understand the results. For other exercises, such as decision trees and ensembles, accuracy scores are reported to evaluate the performance of the models.
