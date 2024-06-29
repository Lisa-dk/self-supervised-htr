# Self-Supervised Hanwritten Text Recognition Using General Adversarial Networks

This repository includes the implementation of the models and experiments used in the thesis 'Towards Self-supervised Handwritting Text Recognition using General Adversarial Networks'.

In this thesis, a self-supervised HTR framework was proposed and investigated. The framework is based on image similarity, where an image generator produces images of the HTR-predicted texts and a loss function computes the dissimilarity between the input and synthetic images. This was first tested for Handwritten Character Recognition, for which the implementation is available at https://github.com/Lisa-dk/self-supervised-mnist.git. 

This repository then contains the following:
- A PyTorch implementation of Puigcerver's HTR model, adapted with column-wise max-pooling. For the model architecture, see "[Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?](https://ieeexplore.ieee.org/document/8269951)".
- The [implementation](https://github.com/omni-us/research-GANwriting) of [GANwriting](https://link.springer.com/chapter/10.1007/978-3-030-58592-1_17), adapted s.t. the generator is conditioned on one-hot encoded labels, the text and images are encoded and processed in the same manner as the HTR model's chosen preprocessing.
- The implementation of the self-supervised HTR framework with two image-based losses and two style-invariant losses.
- An implementation of a Siamese Network used for one of the style-invariant losses.
- The code for the creation of the new datasplits (and their recreation with synthetic images): IAM-GEN, IAM-HTR, IAM-GEN-SIA as described in the thesis.

The structure is based on the implementation of [HTR-Flor](https://ieeexplore.ieee.org/document/9266005)
