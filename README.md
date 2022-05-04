# A PyTorch implementation of MatchboxNet from Scratch

An implementation of the [MatchboxNet: 1D Time-Channel Separable Convolutional Neural Network Architecture for Speech Commands Recognition](https://arxiv.org/abs/2004.08531) architecture. No training logic, only the model (i.e., only a forward pass through the untrained model is possible at the moment).

## Using the model
```
from MatchboxNet import MatchboxNet
import torch

model = MatchboxNet(B=3, R=2, C=64, NUM_CLASSES=30)

sample_mfcc_features = torch.randn(1, 64, 465)
predictions = model(sample_mfcc_features)

print(predictions.shape)
```
