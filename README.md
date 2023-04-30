##  Revealing example of a self-attention model, the building block of a transformer model

Have you heard of transformer models and attempted to study their code but struggled to locate the essential components that make them tick? Look no further, you've arrived at the right place!

This GitHub repository presents a lucid, minimalistic example of an attention mechanism, which forms the backbone of a transformer model. By simplifying the code, the self-attention model becomes easier to grasp. To examine the code, open `model_and_script.py` and navigate to line 20, where you'll find a straightforward self-attention model implementation.

Inside `SelfAttentionModel`, the code proceeds to demonstrate the model's execution process starting at line 36. Employing a loop, each sample in the batch is sequentially processed through self-attention. Within this loop, queries, keys, and values are computed, and attention weights are calculated using queries, keys, and the softmax function. Following this, attention is applied to the values, yielding the self-attention mechanism's output. At line 43, the self-attention results from individual samples are aggregated into a batch. Finally, the self-attention output is passed through an additional layer to produce the model's final output. Tensor shapes are provided after each line of code to clarify data manipulation.

The model is tested on the widely-recognized MNIST dataset. To maintain simplicity, the 28x28 pixels of each image are flattened into 784 inputs. Although the results may not be as remarkable as those of other deep learning models, the primary goal is to offer the most straightforward, concise example of self-attention. The model's speed is also limited due to the use of a for-loop for processing samples sequentially. Employing built-in functions would enable parallel computation of self-attention for each sample, enhancing performance.

To run the code, simply execute python3 model_and_script.py, or you can refer to results.txt.

With this foundational understanding of self-attention, you're now equipped with the essential knowledge to comprehend how transformer models function. Your next step is to explore further resources on constructing transformer models based on self-attention mechanisms. Good luck!

## Issues

Please feel free to reach out if you encounter any issues or even have suspicions about potential problems within this repository. Your feedback is greatly appreciated and will help ensure the accuracy and quality of the content presented.

## Acknowledgements

* Thanks to [GaggiX](https://news.ycombinator.com/user?id=GaggiX) for helping identify issues with this repository.
* Thanks to [James Stevenson](https://github.com/jminuse) for identifying and fixing a memory leak issue.

## Requirements
* [Python3](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [TorchMetrics](https://torchmetrics.readthedocs.io/)
* Linux Environment (Recommended)

## Download
* Download: [zip](https://github.com/jostmey/NakedAttention/zipball/master)
* Git: `git clone https://github.com/jostmey/NakedAttention`
