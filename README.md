# Probabilistic Deep Learning

## Introduction

In this project, I am exploring the use of probabilistic deep learning models, in particular Variational Autoencoders, for the task of image generation.
I will be using several different VAEs to generate images of digits from the MNIST dataset.
The models will be trained using PyTorch and the results will be evaluated using the ELBO loss and the log-likelihood of the test data.

## Installation

Make sure you have a NVIDIA GPU with CUDA installed.

To prepare the environment, run the following commands:

```bash
docker compose up
```

This will build a Docker image with all the necessary dependencies and start a container with Jupyter Lab running on port 8888.

## Acknowledgements

This project is an adaptation from my previous course project in the course [Probabilistic Machine Learning](https://kurser.ku.dk/course/NDAK21004U/2023-2024) at University of Copenhagen.

See some of the original work [here](https://github.com/Minhao-Zhang/PML_Final_2023_B2).
