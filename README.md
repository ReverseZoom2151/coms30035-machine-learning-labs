# COMS30035 - Machine Learning 

This unit covers various Machine Learning (ML) exercises and implementations, including:

1. **Principal Component Analysis (PCA) on the MNIST dataset:** PCA is applied to the MNIST dataset of handwritten digits to reduce the dimensionality of the data while preserving as much variance as possible. The data is then reconstructed using the first K components and visualize the results.
2. **Independent Component Analysis (ICA) on mixed images:** ICA is used to separate three individual sources from a mixed signal, represented by three images. The results of ICA are compared to traditional PCA and show that ICA is able to separate the sources more effectively.
3. **Decision Trees and Ensembles on classification datasets:** Decision tree classifiers are trained on various classification datasets and evaluate their performance. Bagging and random forests are then implemented to improve the accuracy and diversity of the models.
4. **Hidden Markov Models (HMM) for part-of-speech tagging:** HMMs are used to perform part-of-speech tagging on the Brown corpus dataset. The emission and transition models are both implemented for the HMM and use the Viterbi algorithm to decode the tags for a given sequence of words.
5. **Gaussian Mixture Models (GMM) for image segmentation:** GMMs are used to perform image segmentation on a synthetic dataset. The expectation-maximization algorithm is implemented to estimate the parameters of the GMM and use the resulting model to segment the images.
6. **Neural Networks for digit recognition:** A neural network is trained to recognize handwritten digits using the MNIST dataset. A simple feedforward neural network with one hidden layer is implemented and backpropagation is used to optimize the weights.
7. **Bayesian Inference for linear regression:** Bayesian inference is used to perform linear regression on a synthetic dataset. Prior distributions for the model parameters are defined and Markov chain Monte Carlo (MCMC) is used to sample from the posterior distribution.

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

You can install these packages using either `pip` or `conda`. Here are the commands to install the packages using both methods:

### Pip

`pip install numpy matplotlib scikit-learn nltk pymc3 tensorflow`

### Conda

`conda install numpy matplotlib scikit-learn nltk pymc3 tensorflow`

Note that you may need to create a new Conda environment to install the packages, as some packages may have dependencies that are not compatible with your current environment. To create a new Conda environment, you can use the following command:

`conda create -n ml-project python=3.x numpy matplotlib scikit-learn nltk pymc3 tensorflow`

This will create a new Conda environment called ml-project with Python 3.x and the required packages installed. To activate the environment, you can use the following command:

`conda activate ml-project`

## Results
The results of each exercise are presented in the corresponding notebook. For some exercises, such as PCA and ICA, visualizations are provided to help understand the results. For other exercises, such as decision trees and ensembles, accuracy scores are reported to evaluate the performance of the models.
