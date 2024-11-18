# GAN Model with Custom Loss and FID Calculation

This project implements a Generative Adversarial Network (GAN) with a custom loss function and calculates the Fréchet Inception Distance (FID) using a custom encoder model. The goal is to generate images from gene expression data.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Parameters](#parameters)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Installation

1. Create a conda environment and activate it
    ```sh
    conda create --name gan_model python=3.8
    conda activate gan_model
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Project Structure

- `main.py`: Main script to run the training and evaluation.
- `gan/dcgan.py`: Contains the DCGAN model and training loop.
- `gan/wgan.py`: Contains the WGAN model and training loop.
- `gan/early_stopping.py`: Implements early stopping, FID calculation and MS_SSIM calculation.
- `gan/encoder_model.py`: Defines the encoder model.
- `gan/model_config.py`: Configuration parameters for the model.
- `data/`: Directory containing training and testing data.

### Parameters

The parameters for the model are defined in `gan/model_config.py`. Key parameters include:

- `batch_size`: Batch size during training.
- `image_size`: Size of training images.
- `nz`: Size of the latent vector.
- `num_epochs`: Number of training epochs.
- `lr`: Learning rate for optimizers.
- `dataroot_train`: Path to the training dataset.
- `dataroot_test`: Path to the testing dataset.
- `gene_expression_data`: Path to the gene expression data for training.
- `gene_expression_test`: Path to the gene expression data for testing.

### Training

To train the model, run the following command:
```sh
python main.py
```

### Evaluation

The model is evaluated every 10 epochs using the FID and MS-SSIM scores. The best model based on these scores is saved in the `result/model` directory.

### Results

Generated images and model checkpoints are saved in the `result` directory. The FID and MS-SSIM scores are printed during training.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.