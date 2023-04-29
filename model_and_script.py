#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2023-04-28
# Reference: https://peterbloem.nl/blog/transformers
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import dataplumbing_debug as dp
import torch
import torchmetrics

##########################################################################################
# Model
##########################################################################################

class SelfAttentionModel(torch.nn.Module):
  def __init__(self, num_inputs, num_channels, num_outputs, **kwargs):
    super().__init__(**kwargs)

    # Initialize components for self-attention
    #
    self.K = torch.nn.Parameter((2.0*torch.rand(num_inputs, num_inputs)-1.0)/num_inputs**0.5) # Randomly intialize each weight uniformly from [ -1/num_inputs**0.5, 1/num_inputs**0.5 ]
    self.Q = torch.nn.Parameter((2.0*torch.rand(num_inputs, num_inputs)-1.0)/num_inputs**0.5)
    self.V = torch.nn.Parameter((2.0*torch.rand(num_inputs, num_inputs)-1.0)/num_inputs**0.5)

    self.softmax = torch.nn.Softmax(dim=1)

    # Initialize output layer
    #
    self.out = torch.nn.Linear(num_inputs*num_channels, num_outputs)

  def forward(self, x):

    batch_size, num_inputs, num_channels = x.shape

    # Run self attention
    #
    y = []
    for i in range(batch_size): # Process one sample at a time

      x_i = x[i,:,:] # x_i has shape of [ num_inputs, num_channels ]

      x_k_i = torch.matmul(self.K, x_i) # x_k_i has shape of [ num_inputs, num_channels ]
      x_q_i = torch.matmul(self.Q, x_i) # x_q_i has shape of [ num_inputs, num_channels ]
      x_v_i = torch.matmul(self.V, x_i) # x_v_i has shape of [ num_inputs, num_channels ]

      w_i = self.softmax(torch.matmul(x_q_i, x_k_i.T)/num_inputs**0.5) # w_i has shape of [ num_inputs, num_inputs ]
      y_i = torch.matmul(w_i, x_v_i) # y_i has shape of [ num_inputs, num_channels ]

      y.append(y_i)
    y = torch.stack(y, axis=0) # y has shape of [ batch_size, num_inputs, num_channels ]

    # Flatten output
    #
    y_flat = y.reshape([ batch_size, num_inputs*num_channels ]) # y_flat has shape of [ batch_size, num_inputs*num_channels ]

    # Run output layer
    #
    l = self.out(y_flat) # l has shape of [ batch_size, num_outputs ]

    return l

##########################################################################################
# Instantiate model, performance metrics, and optimizer.
##########################################################################################

model = SelfAttentionModel(num_inputs=28**2, num_channels=1, num_outputs=10)
probability = torch.nn.Softmax(dim=1)

loss = torch.nn.CrossEntropyLoss()
accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=10)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

##########################################################################################
# Dataset and data sampler
##########################################################################################

xs_train, ys_train, xs_val, ys_val, xs_test, ys_test = dp.load_mnist(seed=46525)

dataset_train = torch.utils.data.TensorDataset(xs_train, ys_train)
sampler_train = torch.utils.data.RandomSampler(dataset_train, replacement=True)
loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=16, sampler=sampler_train, drop_last=True)

##########################################################################################
# Model
##########################################################################################

i_better = -1
e_better = 1.0e8
a_better = 0.0
state_better = {}

# Loop over the dataset for many epochs
#
for i in range(128):

  # Train the model
  #
  model.train()
  e_train = 0.0
  a_train = 0.0
  for xs_batch, ys_batch in iter(loader_train): # Must use `iter` or `enumerate` for efficiency
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


