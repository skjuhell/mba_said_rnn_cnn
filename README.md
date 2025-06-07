# Deep Learning with R: CNNs, RNNs, and Transformers

This repository contains hands-on examples for students learning to apply **Convolutional Neural Networks (CNNs)**, **Recurrent Neural Networks (RNNs)**, and **Transformers** using R.

You’ll find code for both **regression** and **classification** tasks built with the [R `keras`](https://keras.rstudio.com/) and [`torch`](https://torch.mlverse.org/) packages.

---

## Contents

### CNN Tasks
- `cnn_classification.R` – image classification using CNNs (e.g., MNIST)
- `cnn_regression.R` – CNN-based regression task (e.g., predicting coordinates from images)

### RNN Tasks
- `rnn_classification.R` – sequence classification using simple RNNs
- `rnn_regression.R` – time series prediction using LSTM

### Transformers
- see the Transformers Example from the TensorFlow for R site:  
[tensorflow.rstudio.com/examples](https://tensorflow.rstudio.com/examples/?_gl=1*139omhb*_up*MQ..*_ga*MjEyODM1NTUzMC4xNzQ5MzI3NjY3*_ga_X64JZVV9NC*czE3NDkzMjc2NjYkbzEkZzAkdDE3NDkzMjc2NjYkajYwJGwwJGgw)

---

## How to use it

- Each script is annotated and runnable with minimal setup.
- Tasks are aligned with common course topics (e.g., time series, image classification).
- You can experiment by modifying layer sizes, learning rates, or architectures.

---

## Getting Started

### Prerequisites
- R ≥ 4.2
- Install the required packages:
```r
install.packages(c("keras", "torch", "ggplot2"))
library(keras)
install_keras()
library(torch)
