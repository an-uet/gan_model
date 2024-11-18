# Wgan-div implementation from paper: https://arxiv.org/pdf/1701.07875
import logging
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torchvision.utils as vutils

from gan.model_config import gene_expression_data, noise_scale, batch_size, nz, gene_expression_test, random_indices, \
    nc, lr_g, lr_d, beta1, num_epochs, log_interval, n_critic, image_size, ngpu, lambda_gp
from gan.utils import Utils


# Configure logging
logging.basicConfig(filename='result/training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

img_shape = (nc, image_size, image_size)

k = 2
p = 4




def run():
    losses_d = []
    losses_g = []
    real_grad_arr = []
    fake_grad_arr = []
    best_g_loss = float('inf')
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if cuda else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    gx_train = Utils.load_gene_expression(gene_expression_data, noise_scale)
    image_paths = [f'data/image/spots_train/train/{i}.jpg' for i in range(len(gx_train))]
    dataloader = Utils.load_pair_gx_image(gx_train, image_paths)

    # create a fixed noise vector for visualization of the generator output
    gx_test = Utils.load_gene_expression(gene_expression_test, noise_scale)
    image_fixed = [f'data/image/spots_test/test/{i}.jpg' for i in random_indices]
    image_tensors = []
    for img_path in image_fixed:
        img = Image.open(img_path)
        img_tensor = ToTensor()(img)
        image_tensors.append(img_tensor)
    real_imgs_fixed = torch.stack(image_tensors).to(device)

    gene_expression_fixed = np.array([gx_test[i] for i in random_indices])
    tensor = torch.tensor(gene_expression_fixed, dtype=torch.float32)
    tensor = tensor.unsqueeze(2).unsqueeze(3)
    fixed_noise = tensor.to(device)

    # Initialize generator and discriminator
    generator = Generator(ngpu).to(device)
    discriminator = Discriminator(ngpu).to(device)

    if cuda and ngpu > 1:
        generator = nn.DataParallel(generator, list(range(ngpu)))
        discriminator = nn.DataParallel(discriminator, list(range(ngpu)))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))

    batches_done = 0
    for epoch in range(1, num_epochs + 1):
        for i, (gx_batch, imgs) in enumerate(dataloader):
            real_imgs = Variable(imgs.type(Tensor), requires_grad=True)

            optimizer_D.zero_grad()
            tensor_gene = gx_batch.clone().detach().float()
            tensor_gene = tensor_gene.unsqueeze(2).unsqueeze(3)
            noise = tensor_gene.to(device)

            random_noise = torch.randn(imgs.shape[0], nz, 1, 1, device=device)
            min_noise, max_noise = random_noise.min(), random_noise.max()
            scaled_noise = 2 * (random_noise - min_noise) / (max_noise - min_noise) - 1
            scaled_noise = scaled_noise.to(device)
            z = noise + noise_scale * scaled_noise
            z = z.view(z.size(0), -1)

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)

            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % n_critic == 0:
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()
                # wandb.log({
                #     "D loss": d_loss.item(),
                #     "G loss": g_loss.item(),
                #     "D grad norm": real_grad.norm().item(),
                #     "G grad norm": fake_grad.norm().item()
                # })

                logging.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [gradient penalty: %f]"
                    % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), gradient_penalty.item())
                )
                losses_d.append(d_loss.item())
                losses_g.append(g_loss.item())
                Utils.plot_losses(losses_d, losses_g, 'result/generated_image/losses.png')
                batches_done += n_critic

                if g_loss.item() < best_g_loss:
                    best_g_loss = g_loss.item()
                    torch.save(generator.state_dict(), 'result/model/best_generator.pth')
                    logging.info(f"Best model saved with G loss: {best_g_loss}")

        if epoch % 2 == 0 and (i == len(dataloader) - 1):
            fake_imgs_fixed = generator(fixed_noise.view(fixed_noise.size(0), -1))
            stacked_images = torch.stack((real_imgs_fixed[:32], fake_imgs_fixed[:32]), dim=1)
            mixed_images = stacked_images.view(-1, *real_imgs_fixed.shape[1:])
            mixed_images = F.interpolate(mixed_images, size=(256, 256), mode='bilinear', align_corners=False)
            grid = vutils.make_grid(mixed_images, padding=2, normalize=True, scale_each=True)
            save_image(grid, f"result/generated_image/{epoch}.png", nrow=5, normalize=True)
            # wandb.log({"Generated Images": wandb.Image(grid, caption="Epoch %d" % epoch)})


        if epoch % 50 == 0 and epoch >= 300:
            torch.save(generator.state_dict(), f"result/model/model_epoch_{epoch}.h5")

    data_loss = []
    for loss_d, loss_g, real_grad, fake_grad in zip(losses_d, losses_g, real_grad_arr, fake_grad_arr):
        data_loss.append((loss_d, loss_g, real_grad, fake_grad))
    data_loss = pd.DataFrame(data_loss, columns=['loss_d', 'loss_g', 'real_grad', 'fake_grad'])
    data_loss.to_csv('result/model/losses.csv')


def predict_image(model_path, gene_expression):
    # Load model
    netG = Generator(ngpu)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Remove 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    netG.load_state_dict(new_state_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG.to(device)
    netG.eval()

    noise = np.array(gene_expression)
    noise = torch.tensor(noise, dtype=torch.float32)
    noise = noise.unsqueeze(2).unsqueeze(3)
    noise = noise.to(device)

    random_noise = torch.randn(len(gene_expression), nz, 1, 1, device=device)
    min_noise, max_noise = random_noise.min(), random_noise.max()
    scaled_noise = 2 * (random_noise - min_noise) / (max_noise - min_noise) - 1
    scaled_noise = scaled_noise.to(device)
    noise = noise + 0 * scaled_noise

    full_image = [f'/home/anlt69/Downloads/GAN_model_13Nov2024/gan_model/data/image_1/spots_validation/validation/{i}.jpg' for i in range(500)]

    full_image_tensors = []
    for img_path in full_image:
        img = Image.open(img_path)
        img_tensor = ToTensor()(img)  # Convert to tensor
        full_image_tensors.append(img_tensor)

    with torch.no_grad():
        fake_full = netG(noise).detach().cpu()

    for i, (real_img, fake_img) in enumerate(zip(full_image_tensors, fake_full)):
        fake_img = (fake_img + 1) / 2.0  # Rescale to [0, 1]
        fake_img_np = fake_img.permute(1, 2, 0).numpy()
        real_img_np = real_img.permute(1, 2, 0).numpy()

        # Concatenate real and fake images
        concat_img = np.concatenate((real_img_np, fake_img_np), axis=1)
        concat_img = fake_img_np
        plt.imsave(f'result/validation/image_{i}.jpg', concat_img)



def load_model(model_path, ngpu):
    netG = Generator(ngpu)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Remove 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    netG.load_state_dict(new_state_dict)
    return netG

def predict_and_concat(models, gene_expression, real_image_path, device, noise_scale, nz):
    # Load real image
    real_img = Image.open(real_image_path)
    real_img_tensor = ToTensor()(real_img).to(device)

    # Prepare gene expression noise
    noise = np.array(gene_expression)
    noise = torch.tensor(noise, dtype=torch.float32)
    if noise.dim() == 1:
        noise = noise.unsqueeze(0)
    noise = torch.tensor(noise, dtype=torch.float32).unsqueeze(2).unsqueeze(3).to(device)

    # Generate random noise
    random_noise = torch.randn(1, nz, 1, 1, device=device)
    min_noise, max_noise = random_noise.min(), random_noise.max()
    scaled_noise = 2 * (random_noise - min_noise) / (max_noise - min_noise) - 1
    scaled_noise = scaled_noise.to(device)
    # noise = noise + noise_scale * scaled_noise

    real_img_np = real_img_tensor.permute(1, 2, 0).cpu().numpy()
    list_fake = []
    # Predict fake images and concatenate with real image
    for i, model_path in enumerate(models):
        netG = load_model(model_path, ngpu)
        netG.to(device)
        netG.eval()

        with torch.no_grad():
            fake_img = netG(noise).detach().cpu().squeeze(0)
            fake_img = (fake_img + 1) / 2.0  # Rescale to [0, 1]
            fake_img_np = fake_img.permute(1, 2, 0).numpy()


            # Concatenate real and fake images
            list_fake.append(fake_img_np)
    concat_img = np.concatenate([real_img_np] + list_fake, axis=1)
    plt.imsave(f'result/concat_image.jpg', concat_img)

