# Generative Adversarial Network (GAN) for Image Generation 
This project utilizes a Generative Adversarial Network (GAN) to generate synthetic flower images. The GAN is trained on a dataset of real flower images, allowing it to learn the patterns and features of the images and generate new, realistic images. 

## Project structure
- `train.py` — Trains the DCGAN and saves the checkpoint.
- `src/models.py` — Generator and Discriminator models.
- `src/data.py` — Loads the dataset.
- `src/visualization.py` — Plotting helpers.
- `notebooks/demo.ipynb` — Loads saved model and outputs and reproduces the results. 


## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
Use the official Oxford VGG dataset page to load the archive with images: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ 

The training code expects:

* 102flowers.tgz 

* category_to_images.json 



## Usage

Train
Run training (defaults are defined in train.py):
```bash
python train.py
```

Or run with custom settings:

```bash
python train.py \
  --tgz-path 102flowers.tgz \
  --json-path category_to_images.json \
  --epochs 50 \
  --batch-size 64 \
  --lr 2e-4 \
  --seed 0 \
  --out-dir outputs/run1
```
