#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2023-04-28
# Reference: https://peterbloem.nl/blog/transformers
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import dataplumbing as dp
import torch
import torchmetrics

##########################################################################################
# Settings
##########################################################################################

# Settings for training
#
batch = 1024
step = 0.1
epochs = 128

# Setings for model
#
width = 32

# Settings for environment
#
device = torch.device('cpu')
seed = 46525

##########################################################################################
# Model
##########################################################################################

class LinearWithNorm(torch.nn.Module):
  def __init__(self, num_inputs, num_outputs, **kwargs):
    super().__init__(**kwargs)

    # Initialize linear layer with batch normalization
    #
    self.linear = torch.nn.Linear(num_inputs, num_outputs)
    self.norm = torch.nn.BatchNorm1d(num_outputs)

  def forward(self, x):

    # Run linear layer with batch normalization
    #
    l = self.linear(x)
    n = self.norm(l)

    return n

class SelfAttentionModel(torch.nn.Module):
  def __init__(self, num_inputs, num_outputs, width=128, **kwargs):
    super().__init__(**kwargs)

    # Initialize transformer module
    #
    self.key = LinearWithNorm(num_inputs, width) # `LinearWithNorm` combines torch.nn.Linear and torch.nn.BatchNorm1d
    self.query = LinearWithNorm(num_inputs, width)
    self.value = LinearWithNorm(num_inputs, width)
    self.softmax = torch.nn.Softmax(dim=2)

    # Initialize output layer
    #
    self.out = LinearWithNorm(width, num_outputs) # `LinearWithNorm` combines torch.nn.Linear and torch.nn.BatchNorm1d

  def forward(self, x):

    # Run transformer module
    #
    ks = self.key(x) # Shape of [ batch_size, width ]
    qs = self.query(x) # Shape of [ batch_size, width ]
    vs = self.value(x) # Shape of [ batch_size, width ]

    batch_size, width = ks.shape
    ks_ = ks.reshape([ batch_size, 1, width ]) # Shape of [ batch_size, 1, width ]
    qs_ = qs.reshape([ batch_size, width, 1 ]) # Shape of [ batch_size, width, 1 ]
    vs_ = vs.reshape([ batch_size, 1, width ]) # Shape of [ batch_size, 1, width ]

    ws_ = self.softmax(ks_*qs_/width**0.5) # Shape of [ batch_size, width, width ]. The softmax ensures the sum of values along the last dimension is always 1.
    ys = torch.sum(ws_*vs_, axis=2) # Shape of [ batch_size, width ]

    # Run output layer
    #
    ls = self.out(ys) # Shape of [ batch_size, 10 ]

    return ls

##########################################################################################
# Instantiate model, performance metrics, and optimizer.
##########################################################################################

model = SelfAttentionModel(28**2, 10, width=32).to(device)
probability = torch.nn.Softmax(dim=1).to(device)

loss = torch.nn.CrossEntropyLoss()
accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=10).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=step)

##########################################################################################
# Load data
##########################################################################################

xs_train, ys_train, xs_val, ys_val, xs_test, ys_test = dp.load_mnist(seed=seed, device=device)

##########################################################################################
# Data sampling
##########################################################################################

dataset_train = torch.utils.data.TensorDataset(xs_train, ys_train)
sampler_train = torch.utils.data.RandomSampler(dataset_train, replacement=True)
loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch, sampler=sampler_train, drop_last=True)

##########################################################################################
# Model
##########################################################################################

i_better = -1
e_better = 1.0e8
a_better = 0.0
state_better = {}

# Loop over the dataset for many epochs
#
for i in range(epochs):

  # Train the model
  #
  model.train()
  e_train = 0.0
  a_train = 0.0
  for j, (xs_batch, ys_batch) in enumerate(loader_train):
    ls_batch = model(xs_batch)
    ps_batch = probability(ls_batch) # Model outputs logits that we must convert to probabilities
    e_batch = loss(ls_batch, ys_batch) # CrossEntropyLoss requires logits
    a_batch = accuracy(ps_batch, ys_batch)
    optimizer.zero_grad()
    e_batch.backward()
    optimizer.step()
    e_train += e_batch/len(loader_train) # Accumulate average loss for this epoch
    a_train += a_batch/len(loader_train) # Accumulate average accuracy for this epoch

  # Assess performance on validation data
  #
  model.eval()
  with torch.no_grad():
    ls_val = model(xs_val)
    ps_val = probability(ls_val) # Model outputs logits that we must convert to probabilities
    e_val = loss(ls_val, ys_val) # CrossEntropyLoss requires logits
    a_val = accuracy(ps_val, ys_val)
    if e_val < e_better: # Early stopping check
      i_better = i
      e_better = e_val
      a_better = a_val
      state_better = model.state_dict()

  # Print report
  #
  print(
    'i: '+str(i),
    'e_train: {:.5f}'.format(float(e_train)/0.693)+' bits',
    'a_train: {:.1f}'.format(100.0*float(a_train))+' %',
    'e_val: {:.5f}'.format(float(e_val)/0.693)+' bits',
    'a_val: {:.1f}'.format(100.0*float(a_val))+' %',
    sep='\t', flush=True
  )

model.eval()
model.load_state_dict(state_better)
with torch.no_grad():
  ls_test = model(xs_test)
  ps_test = probability(ls_test) # Model outputs logits that we must convert to probabilities
  e_test = loss(ls_test, ys_test) # CrossEntropyLoss requires logits
  a_test = accuracy(ps_test, ys_test)

print(
  'e_test: {:.5f}'.format(float(e_test)/0.693)+' bits',
  'a_test: {:.1f}'.format(100.0*float(a_test))+' %',
  sep='\t', flush=True
)


