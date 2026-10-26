# MLP, Step by Step.

**An Interactive, First-Principles Exploration of Multilayer Perceptrons.**

[**Live Interactive Demo**](https://chandraprakash-bathula.github.io/MLP/)

## Overview

**MLP, Step by Step** is an interactive educational project that explores how a multilayer perceptron (MLP) works, learns, and makes predictions. The project connects mathematical foundations with practical neural network computation through interactive visualizations, step-by-step derivations, and live model calculations.

Using the UCI Heart Disease dataset from Cleveland, the project follows individual patient records through a neural network and explains how input features are transformed into predictions. It explores the mathematical operations behind forward propagation, activation functions, loss computation, backpropagation, gradient descent, optimization, and model evaluation.

Rather than treating neural networks as black boxes, this project aims to make their internal computations transparent and interpretable.

## Project Objectives

- Understand the mathematical foundations of multilayer perceptrons.
- Explore how neurons, weights, biases, and activation functions work together.
- Derive forward propagation and backpropagation step by step.
- Understand parameter initialization and gradient-based optimization.
- Investigate batch normalization and dropout.
- Connect loss functions to probabilistic modeling.
- Explore how optimization algorithms update network parameters.
- Understand model evaluation using real-world clinical data.
- Bridge theoretical neural network concepts with practical machine learning workflows.

## Interactive Learning Experience

Explore the project through the live website.

**[Launch MLP, Step by Step](https://chandraprakash-bathula.github.io/MLP/).**

The interactive experience includes two major parts.

### Part I. Foundations.

Explore the ideas that led to modern neural networks and understand why multilayer architectures are necessary.

Key topics include:

- A brief history of deep learning.
- Biological neurons and artificial neurons.
- Perceptrons and logistic regression.
- The limitations of single-layer models.
- Hidden layers and nonlinear decision boundaries.
- The applied machine learning pipeline.

### Part II. The Network, Derived.

Follow the mathematical operations performed by a multilayer perceptron, from input preprocessing to final evaluation.

Key topics include:

- Mathematical notation and the learning problem.
- Network architecture and parameter initialization.
- Neurons, activation functions, and nonlinear transformations.
- Batch normalization and its backward pass.
- Dropout and regularization.
- Likelihood, cross-entropy, and loss computation.
- Backpropagation and reverse-mode differentiation.
- Gradient descent and parameter updates.
- Optimization algorithms.
- Training, regularization, and generalization.
- Model evaluation and performance analysis.

## Network Architecture

The interactive example uses a five-layer multilayer perceptron with the following architecture.

**Network Architecture:** `13 → 16 → 12 → 8 → 4 → 1`.

| Component | Description |
|---|---|
| Input layer | 13 standardized clinical features. |
| Hidden layer 1 | 16 neurons. |
| Hidden layer 2 | 12 neurons. |
| Hidden layer 3 | 8 neurons. |
| Hidden layer 4 | 4 neurons. |
| Output layer | 1 sigmoid output for binary prediction. |

The network demonstrates how information propagates through successive affine transformations and nonlinear activation functions. During training, backpropagation computes gradients that guide updates to the network's weights and biases.

## Dataset

The project uses the **UCI Heart Disease dataset, Cleveland subset**, containing 303 patient records and 13 input features.

The interactive example uses a training and test split.

- **Dataset:** UCI Heart Disease, Cleveland subset.
- **Total records:** 303.
- **Input features:** 13.
- **Training records:** 242.
- **Held-out test records:** 61.
- **Task:** Binary classification.

The dataset provides a practical context for studying how neural networks learn patterns from structured tabular data.

Dataset resource: [UCI Machine Learning Repository, Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease).

This is an educational demonstration, not a clinically validated diagnostic system.

## Core Mathematical Concepts

### 1. Forward Propagation.

Each layer transforms its input using a weight matrix, a bias vector, and an activation function.

For layer \(l\), the linear transformation is:

$$
z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}
$$

The activation is then calculated as:

$$
a^{(l)} = \phi^{(l)}\left(z^{(l)}\right)
$$

Here, \(W^{(l)}\) represents the weight matrix, \(b^{(l)}\) represents the bias vector, and \(\phi^{(l)}\) denotes the activation function for layer \(l\).

These operations repeat across the network until the output layer produces a prediction.

### 2. Activation Functions.

Activation functions introduce nonlinear transformations that allow neural networks to learn complex patterns.

For example, the ReLU activation function is defined as:

$$
\mathrm{ReLU}(x) = \max(0,x)
$$

The sigmoid activation function is defined as:

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

ReLU is commonly used in hidden layers, while sigmoid can transform a binary classification output into a probability between zero and one.

### 3. Weight Initialization.

Appropriate initialization helps maintain stable activation and gradient scales as information passes through multiple layers.

Xavier initialization scales weights according to the number of input and output units. He initialization is commonly used with ReLU activations.

The corresponding variance relationships are:

$$
\mathrm{Var}(W) = \frac{2}{n_{\text{in}}+n_{\text{out}}}
$$

for Xavier initialization, and:

$$
\mathrm{Var}(W) = \frac{2}{n_{\text{in}}}
$$

for He initialization, where \(n_{\text{in}}\) is the number of input units and \(n_{\text{out}}\) is the number of output units.

### 4. Binary Cross-Entropy Loss.

The loss function quantifies the discrepancy between model predictions and target labels.

For binary classification, binary cross-entropy is defined as:

$$
\mathcal{L} =
-\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i\log(\hat{y}_i)
+
(1-y_i)\log(1-\hat{y}_i)
\right]
$$

Here, \(N\) is the number of samples, \(y_i\) is the true binary label, and \(\hat{y}_i\) is the predicted probability for sample \(i\).

The objective is to minimize the loss by learning suitable network parameters.

### 5. Backpropagation.

Backpropagation applies the chain rule to calculate how the loss changes with respect to each trainable parameter.

For a weight matrix in layer \(l\), the gradient is:

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} =
\frac{\partial \mathcal{L}}{\partial z^{(l)}}
\left(a^{(l-1)}\right)^\top
$$

The gradient indicates how the loss changes when the corresponding weights change. Backpropagation propagates these derivatives through the network to calculate gradients for earlier layers.

### 6. Gradient Descent and Optimization.

Gradient descent updates model parameters in the direction that reduces the loss.

The parameter update rule is:

$$
\theta_{t+1} =
\theta_t -
\eta\nabla_{\theta}\mathcal{L}(\theta_t)
$$

Here, \(\theta_t\) represents the model parameters at step \(t\), \(\eta\) is the learning rate, and \(\nabla_{\theta}\mathcal{L}\) is the gradient of the loss with respect to the parameters.

The learning rate controls the size of each update. Optimization algorithms such as Adam adapt the update process to improve training behavior.

### 7. Regularization.

Regularization techniques help neural networks generalize beyond their training examples.

Dropout randomly disables selected activations during training. For an activation \(a\), inverted dropout can be expressed as:

$$
\tilde{a} = \frac{m \odot a}{1-p}
$$

where \(m\) is a random binary mask, \(p\) is the dropout probability, and \(\odot\) denotes element-wise multiplication.

Batch normalization uses batch statistics to normalize intermediate activations during training. Both techniques can influence training dynamics and generalization, although their effects depend on the model and training configuration.

## Learning Outcomes.

By exploring this project, learners can develop a stronger understanding of the following concepts.

- How artificial neurons perform mathematical computations.
- Why hidden layers enable nonlinear representation learning.
- How forward propagation transforms input features into predictions.
- How initialization influences training dynamics.
- How loss functions connect predictions to learning objectives.
- How backpropagation computes gradients using the chain rule.
- How optimizers update parameters during training.
- How regularization affects generalization.
- How to evaluate neural networks using held-out data.
- How mathematical derivations correspond to actual model computations.

## Technologies and Implementation.

The project combines mathematical explanations, interactive visualizations, and browser-based model calculations.

The live interface includes controls for exploring different held-out patients, activation functions, initialization choices, and mathematical explanations.

Refer to the implementation for the exact technologies, training configuration, and computational details.

## Future Directions.

Potential extensions include the following.

- Comparing MLP performance with logistic regression and other classical machine learning models.
- Investigating the effects of learning rates and optimizer configurations.
- Visualizing gradient flow across layers during training.
- Studying vanishing and exploding gradients.
- Comparing different initialization strategies experimentally.
- Exploring the effects of dropout and batch normalization on generalization.
- Adding calibration analysis and decision-threshold comparisons.
- Extending the interactive framework to other neural network architectures.

## Author.

**Chandra Prakash Bathula.**

Adjunct Faculty, Saint Louis University.

- [GitHub](https://github.com/ChandraPrakash-Bathula).
- [LinkedIn](https://www.linkedin.com/in/chandra-prakash-bathula/).
- [Medium](https://medium.com/@ChandraPrakash-Bathula).

## Academic Use.

This project is intended for educational and research exploration. The examples illustrate neural network concepts and should not be interpreted as medical advice or used as a substitute for professional clinical evaluation.

Users should consult the original dataset documentation and applicable licensing terms when reusing the data.

## References.

- [UCI Machine Learning Repository. Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).
- [Deep Learning Book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/).

---

**MLP, Step by Step.**

*From mathematical foundations to interactive neural network computation.*
