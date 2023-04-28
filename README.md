#  Revealing example of a self-attention model, the building block of a transformer model

Have you heard of transformer models and attempted to study their code but struggled to locate the essential components that make them tick? Look no further, you've arrived at the right place!

This GitHub repository offers a clear, minimalistic example of an attention mechanism - the core building block of a transformer model. By stripping the code down to its simplest form, understanding the self-attention model becomes more accessible. To dive right into the code, open `model_and_script.py` and jump to line 38, where you'll find the code for a straightforward self-attention model.

Within SelfAttentionModel, you'll initially encounter code responsible for instantiating the layers needed to run the model. For the sake of simplicity, a custom layer called `LinearWithNorm` has been created, which combines `torch.nn.Linear` and `torch.nn.BatchNorm1d` into a single class. Utilizing this, we instantiate linear transformations for `queries`, `keys`, and `values`, as well as a linear transformation to convert the self-attention model's output into the necessary format for predictions.

Following this, the code demonstrates how to execute the model. First, queries, keys, and values are computed, and then attention weights are calculated using `queries`, `keys`, and the softmax function. Subsequently, attention is applied to the `values`, which are then passed through a final layer to produce the model's output. To better illustrate data manipulation, tensor shapes are provided after each line of code.

The model is deployed on the widely-used MNIST dataset. In order to maintain simplicity, the 28x28 pixels of each image have been flattened into 784 inputs. Though the results may not be as impressive as those of other deep learning models, the main objective is to provide the simplest, most concise example of self-attention.

To run the code simply execute `python3 model_and_script.py` or you can just look at `results.txt`.

With this foundational understanding of self-attention, you're now equipped with the essential knowledge to comprehend how transformer models function. Your next step is to explore further resources on constructing transformer models based on self-attention mechanisms. Good luck!

## Requirements
* [Python3](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [TorchMetrics](https://torchmetrics.readthedocs.io/)
* Linux Environment (Recommended)

## Download
* Download: [zip](https://github.com/jostmey/NakedAttention/zipball/master)
* Git: `git clone https://github.com/jostmey/NakedAttention`
