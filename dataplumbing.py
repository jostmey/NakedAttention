#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2023-04-28
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import torchvision
import torch

##########################################################################################
# Load data
##########################################################################################

# Load training, validation, and test data from the MNIST dataset
#
def load_mnist(seed=None, device=torch.device('cpu')):

  # Random number generator
  # 
  generator = torch.Generator(device=device)
  if seed is not None:
    generator.manual_seed(seed)

  # Load MNIST dataset
  #
  samples_train = torchvision.datasets.MNIST('./', train=True, download=True)
  samples_test = torchvision.datasets.MNIST('./', train=False, download=True)

  # Format features and labels
  #
  xs = samples_train.data.to(device)
  num = xs.shape[0]
  xs = xs.reshape([ num, 28**2, 1 ])
  xs = xs.type(torch.float32)
  ys = samples_train.train_labels.to(device)

  xs_test = samples_test.data.to(device)
  num_test = xs_test.shape[0]
  xs_test = xs_test.reshape([ num_test, 28**2, 1 ])
  xs_test = xs_test.type(torch.float32)
  ys_test = samples_test.test_labels.to(device)

  # Split into training and validation samples
  #
  num_train = int(num*5/6)
  num_val = num-num_train

  js = torch.randperm(num, generator=generator)
  js_train = js[:num_train]
  js_val = js[num_train:]

  xs_train = xs[js_train]
  ys_train = ys[js_train]

  xs_val = xs[js_val]
  ys_val = ys[js_val]

  # Normalizing features
  #
  mean = torch.mean(xs_train, axis=0, keepdim=True)
  variance = torch.var(xs_train, axis=0, keepdim=True)

  xs_train = (xs_train-mean)/torch.std(variance+1.0E-8)
  xs_val = (xs_val-mean)/torch.std(variance+1.0E-8)
  xs_test = (xs_test-mean)/torch.std(variance+1.0E-8)

  return xs_train, ys_train, xs_val, ys_val, xs_test, ys_test

