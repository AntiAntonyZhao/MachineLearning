# Applied Machine Learning Projects

- [Project 1: Linear Regression and Logistic Regression with Gradient Descent](#project-1-linear-regression-and-logistic-regression-with-gradient-descent)
- [Project 2: Multilayer Perceptron & Convolutional Neural Network (MLP & CNN)](#project-2-multilayer-perceptron--convolutional-neural-network-mlp--cnn)
- [Project 3: Naive Bayes & BERT Models](#project-3-naive-bayes--bert-models)
- [Project 4: Paper Re-implementation - "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"](#project-4-paper-re-implementation---dropout-a-simple-way-to-prevent-neural-networks-from-overfitting)

## Project 1: Linear Regression and Logistic Regression with Gradient Descent
### **Key Concepts:** Regression Analysis, Classification, Gradient Descent.
### **Datasets:**
- **Boston Housing Dataset:** Features 506 data samples, each with 13 real attributes (excluding 'B' for ethical reasons). The target variable is the median value of owner-occupied homes.
- **Wine Dataset:** Comprises 178 data samples, each with 13 attributes, distributed across three classes.
### **Models:**
- Implemented analytical linear regression for the Boston Housing dataset.
- Applied logistic regression with gradient descent to the Wine dataset.
- Used mini-batch stochastic gradient descent for both models.
### **Experiments:**
- Conducted 80/20 train/test splits and reported performance metrics.
- Employed 5-fold cross-validation for more reliable performance metrics.
- Varied training data sizes and minibatch sizes to assess performance impacts.
- Tested linear and logistic regression with different learning rates and parameter configurations.
- Explored Gaussian basis functions for Dataset 1 and compared results with the original feature set.
- Investigated mini-batch stochastic gradient descent versus analytical solutions.

## Project 2: Multilayer Perceptron & Convolutional Neural Network (MLP & CNN)
### **Key Concepts:** Neural Network Architecture, Deep Learning, Image Recognition.
### **Datasets:**
- **Fashion MNIST:** Features 60,000 training images and 10,000 testing images of Zalando's article items, each a 28x28 grayscale image labeled from 10 classes.
- **STL10:** Contains 13,000 labeled images from 10 object classes, with 5,000 for training and 8,000 for testing.
### **Models:**
- Developed from scratch, incorporating backpropagation and mini-batch gradient descent (e.g., SGD).
- Utilized Python and Numpy; structured MLP as a class with methods for fitting, predicting, and accuracy evaluation.
### **Experiments:**
- Tested different weight initialization methods.
- Compared MLPs with varying numbers and types of layers.
- Investigated the impact of activation functions, regularization, normalization, and CNN architecture on accuracy.
- Analyzed performance changes with different optimizers and pre-trained models.

## Project 3: Naive Bayes & BERT Models
### **Key Concepts:** Text Classification, Natural Language Processing, Transformer Models.
### **Datasets:**
- **Emotion Dataset from Hugging Face:** Consists of English Twitter messages categorized into six basic emotions: anger, fear, joy, love, sadness, and surprise. 
### **Models:**
- **Naive Bayes Model:** Implemented from scratch using Python and Numpy. Functions included for fitting the model, making predictions, and evaluating accuracy.
- **BERT Model:** Employed a BERT-based model using pre-trained weights. Modified or fine-tuned to compare with the pre-trained version.
### **Experiments:**
- **Performance Comparison:** Naive Bayes and BERT models' performances were compared on the Emotion classification task using accuracy metrics.
- **Attention Matrix Analysis:** Examined the attention matrix in the BERT model for correctly and incorrectly predicted documents.
- **Pretraining Impact Investigation:** Explored the benefits of pretraining on an external corpus for the Emotion prediction task and analyzed performance differences between deep learning and traditional machine learning methods.

## Project 4: Paper Re-implementation - "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- **Objective:** Re-implement the "dropout" technique as described in the research paper, examining its effects on model performance using subsets of the datasets mentioned in the paper.
- **Key Concepts:** Model Generalization, Research Analysis, Regularization.

## Technologies Used
- Python
- PyTorch
- Scikit-learn
