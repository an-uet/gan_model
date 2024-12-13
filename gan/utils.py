# Defining the Training Function
import ast
from math import ceil
from typing import List

from scipy.stats import zscore
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import torchvision.datasets as dset
from torchvision import datasets, transforms


from gan.model_config import dataroot_train, image_size, batch_size, workers

class GeneExpressionImageDataset(Dataset):
    def __init__(self, gene_expressions, image_paths, transform=None):
        """
        Args:
            gene_expressions (list or array): Array-like list of gene expression vectors.
            image_paths (list): List of file paths to corresponding images.
            transform (callable, optional): Transform to apply to images.
        """
        self.gene_expressions = gene_expressions
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.gene_expressions)

    def __getitem__(self, idx):
        gene_expression = self.gene_expressions[idx]

        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)

        gene_expression = torch.tensor(gene_expression, dtype=torch.float32)

        return gene_expression, image


class Utils:
    @staticmethod
    def load_gene_expression(data_path='data.csv', scale=0.3):
        df = pd.read_csv(data_path)
        gene_expression = df['gene_expression']
        new_gene_expression = []

        for i in range(len(gene_expression)):
            try:
                row = ast.literal_eval(gene_expression[i])
                row = zscore(np.array(row))
                new_gene_expression.append(row)

            except Exception as e:
                print('error : ', i,  e)
        gene_expression = np.array(new_gene_expression)


        # # add noise
        # noise = np.random.normal(0, 1, gene_expression.shape)
        # noise_min = noise.min()
        # noise_max = noise.max()
        # scaled_noise = (noise - noise_min) / (noise_max - noise_min)
        #
        # scaled_noise = scaled_noise * 2 - 1
        #
        #
        # gene_expression += scale*scaled_noise
        return gene_expression


    # scale matrix to [-1, 1]
    @staticmethod
    def scale_matrix(matrix, max):
        scaled_matrix = 2 * matrix / max - 1
        return scaled_matrix

    @staticmethod
    def plot_images(images, filename):
        h, w, c = images.shape[1:]
        grid_size = ceil(np.sqrt(images.shape[0]))
        images = (images + 1) / 2. * 255.
        images = images.astype(np.uint8)
        images = (images.reshape(grid_size, grid_size, h, w, c)
                  .transpose(0, 2, 1, 3, 4)
                  .reshape(grid_size * h, grid_size * w, c))
        # plt.figure(figsize=(16, 16))
        plt.imsave(filename, images)
        plt.imshow(images)
        plt.show()

    @staticmethod
    def plot_losses(losses_d, losses_g, filename):
        fig, axes = plt.subplots(1, 2, figsize=(8, 2))
        axes[0].plot(losses_d)
        axes[1].plot(losses_g)
        axes[0].set_title("losses_d")
        axes[1].set_title("losses_g")
        plt.tight_layout()
        plt.savefig(filename)
        # plt.close()

    @staticmethod
    def load_image(dataroot):
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=workers)
        return dataloader

    @staticmethod
    def load_pair_gx_image(gene_expression: List[List[float]], image_paths: List[str]):
        dataset= GeneExpressionImageDataset(gene_expressions=gene_expression, image_paths=image_paths,
                                            transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        return dataloader







