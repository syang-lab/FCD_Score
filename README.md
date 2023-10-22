# FCD_Score
Fréchet CLIP Distance Score: Fréchet CLIP Distance Score calculate Fréchet Distance based on features from CLIP image encoder. \
FCD measure similarity between generated images and the dataset.

# Installation
Install from pip:\
pip install pytorch-fcd\
Requirements:\
ftfy\
regex\
numpy\
pillow\
scipy\
torch>=1.7.1\
torchvision>=0.8.2\
clip 

# Usage
calculate and save statistics for datasest:\
pytorch-fcd --model-name "ViT-B/32" --savepath path_to_image_dataset path_to_save_file

calculate FCD score for image dataset and generated image dataset:\
pytorch-fcd --model-name "ViT-B/32" --paths path_to_image_dataset path_to_generated_image_dataset

# Citing
If you use this repository in your research, consider citing it using the following Bibtex entry:\
@misc{ShanYang2023FCD,\
author={Shan Yang},\
title={{pytorch-fcd: FCD Score for PyTorch}},\
month={Nov},\
year={2023},\
note={Version 1.0.0.dev1},\
howpublished={\url{https://github.com/syang-lab/FCD_Score}} 
\}

# License
This implementation is licensed under the Apache License 2.0.


