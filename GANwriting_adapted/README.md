# Adapted GANwriting

This folder contains the code of GANwriting with the adaptations that were necessary for the generator's integration into the self-supervised HTR framework. 
Changes were made as follows:
- Images are now resized to aspect ratio in preprocessing
- New label encoding
- The generator is conditioned on one-hot encoded text labels.
- Automatic FID computation between and across wids
- New data partitions created with [generate_new_datasplits.ipynb]{GANwriting_adapted/Groundtruth/generate_new_datasplits.ipynb}
