# FCD_Score
Fréchet CLIP Distance Score: Fréchet CLIP Distance Score calculate Fréchet Distance based on features from CLIP image encoder. FCD measure similarity between generated images and the dataset.

# Installation
Install from pip:
pip install pytorch-fcd
Requirements:
ftfy
regex
numpy
pillow
scipy
torch>=1.7.1
torchvision>=0.8.2
clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33

# Usage
calculate and save statistics for datasest
python fcd_score.py --model-name "Vit-B/32" --savepath path_to_image_dataset path_to_save_file

calculate FCD score for dataset and generated image dataset
python fcd_score.py --model-name "Vit-B/32" --paths path_to_image_dataset path_to_generated_image_dataset




