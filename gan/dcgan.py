from __future__ import print_function

# Ignore excessive warnings
import logging
import os
import random  # to set the python random seed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import torch.nn.utils.spectral_norm as spectral_norm

from gan.early_stopping import calculate_MS_SSIM, calculate_FID
from gan.encoder_model import Autoencoder, custom_loss_mse
from gan.model_config import num_predict_image, ngf, nz, nc, ndf, gene_expression_data, num_epochs, no_cuda, ngpu, lr_g, \
    lr_d, \
    beta1, real_label, fake_label, noise_scale, dataroot_train, dataroot_test, gene_expression_test, random_indices, \
    gene_expression_validation, alpha
from gan.utils import Utils

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

logging.basicConfig(filename='result/training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

losses_d = []
losses_g = []
losses_g_bce = []
losses_g_fm = []
d_real = []
d_fake1 = []
d_fake2 = []


# Model Definition

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            # state size. (ngf // 2) x 128 x 128
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.features = nn.Sequential(
            # input is (nc) x 256 x 256
            spectral_norm(nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf // 2) x 128 x 128
            spectral_norm(nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Classifier layer
        self.classifier = nn.Conv2d(ndf * 16 + 1, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Extract features
        features = self.features(input)

        # Minibatch standard deviation
        batch_std = self.minibatch_stddev(features)

        # Concatenate the batch standard deviation feature
        features = torch.cat([features, batch_std], dim=1)

        # Classification
        classification = self.sigmoid(self.classifier(features)).view(-1)
        return classification, features

    def minibatch_stddev(self, features, group_size=4):
        size = features.shape
        group_size = min(group_size, size[0])
        subgroups = size[0] // group_size  # Calculate the number of subgroups

        stddev = features.view(subgroups, group_size, -1, size[2], size[3])
        stddev = torch.std(stddev, dim=1, unbiased=False)

        stddev_mean = stddev.mean(dim=[1, 2, 3], keepdim=True)
        # print(stddev_mean.shape)

        stddev_mean = stddev_mean.repeat(group_size, 1, size[2], size[3])  # Shape: [batch_size, 1, H, W]
        # print(stddev_mean.shape)  # For debugging, check shape

        return stddev_mean.contiguous().view(size[0], 1, size[2], size[3])  # Final shape: [batch_size, 1, H, W]


def generator_loss_with_feature_matching(fake, real, disc):
    # Get the discriminator outputs for both fake and real images
    output_fake, features_fake = disc(fake)
    _, features_real = disc(real)

    # Feature matching loss: L2 loss between real and fake features
    feature_matching_loss = torch.mean((features_real - features_fake) ** 2)

    return feature_matching_loss


def train(gen, disc, device, dataloader, optimizerG, optimizerD, criterion, epoch, iters, noise_scale, real_images,
          fixed_noise):
    gen.train()
    disc.train()

    # # Load gene expression data
    # gene_expression = Utils.load_gene_expression(gene_expression_data, noise_scale)

    # Establish convention for real and fake labels during training (with label smoothing)
    for i, d in enumerate(dataloader, 0):
        gene_expression_batch, data = d
        disc.zero_grad()
        # Format batch
        # real_cpu = data[0].to(device)
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        output, _ = disc(real_cpu)
        output = output.view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        tensor_gene = gene_expression_batch.clone().detach().float()
        tensor_gene = tensor_gene.unsqueeze(2).unsqueeze(3)
        noise = tensor_gene.to(device)

        random_noise = torch.randn(b_size, nz, 1, 1, device=device)
        min_noise, max_noise = random_noise.min(), random_noise.max()
        scaled_noise = 2 * (random_noise - min_noise) / (max_noise - min_noise) - 1
        scaled_noise = scaled_noise.to(device)
        noise = noise + noise_scale * scaled_noise

        fake = gen(noise)
        label.fill_(fake_label)
        output, _ = disc(fake.detach())
        output = output.view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        gen.zero_grad()
        label.fill_(real_label)
        output, _ = disc(fake)
        output = output.view(-1)

        errG1 = criterion(output, label)
        errG2 = generator_loss_with_feature_matching(fake, real_cpu, disc)

        errG = alpha*errG1 + (1-alpha)*errG2

        # errG = custom_loss_mse(encoder_model, device, fake, real_cpu, output, criterion, sim_loss_weight=0.5, label=real_label) + generator_loss_with_feature_matching(fake, real_cpu, disc)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            logging.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                         % (epoch, num_epochs, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            losses_d.append(errD.item())
            losses_g.append(errG.item())
            d_real.append(D_x)
            d_fake1.append(D_G_z1)
            d_fake2.append(D_G_z2)
            losses_g_bce.append(errG1.item())
            losses_g_fm.append(errG2.item())

            if not os.path.exists('result/generated_image'):
                os.makedirs('result/generated_image')

            Utils.plot_losses(losses_d, losses_g, 'result/generated_image/losses.png')
            Utils.plot_losses(losses_g_bce, losses_g_fm, 'result/generated_image/losses_g.png')
            Utils.plot_losses(d_real, d_fake2, 'result/generated_image/losses_g.png')


        if epoch % 2 == 0 and (i == len(dataloader) - 1):
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
                # print(fake.shape)
                if not os.path.exists('result/generated_image'):
                    os.makedirs('result/generated_image')

            stacked_images = torch.stack((real_images, fake), dim=1)

            mixed_images = stacked_images.view(-1, *real_images.shape[1:])
            grid = vutils.make_grid(mixed_images, padding=2, normalize=True, scale_each=True)
            save_image(grid, f'result/generated_image/generated_img_epoch_{epoch}.png')

        iters += 1


def main():
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Set random seeds
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.backends.cudnn.deterministic = True

    # Load the saved autoencoder model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = Autoencoder(input_channels=3, encoding_dim=2048)
    autoencoder.load_state_dict(torch.load('encoder_model/encoder_model.pth', map_location=device))
    autoencoder = autoencoder.to(device)
    encoder = autoencoder.encoder

    # Load the dataset
    # trainloader = Utils.load_image(dataroot=dataroot_train)
    testloader = Utils.load_image(dataroot=dataroot_test)

    netG = Generator(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)

    netD = Discriminator(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

    iters = 0
    best_g_loss = float('inf')
    tmp_g_loss = float('inf')
    best_ms_ssim_score = float('inf')
    best_fid_score = float('inf')
    no_improvement_count = 0

    gx_test = Utils.load_gene_expression(gene_expression_test, noise_scale)

    gx_train = Utils.load_gene_expression(gene_expression_data, noise_scale)
    image_paths = [f'data/image/spots_train/train/{i}.jpg' for i in range(len(gx_train))]
    trainloader = Utils.load_pair_gx_image(gx_train, image_paths)

    for epoch in range(1, num_epochs + 1):
        # every 10 epochs, check MS-SSIM and FID scores, if the scores are better than the best scores, save the model
        if epoch % 10 == 0:
            fid_score = calculate_FID(netG, testloader, encoder, gx_test, device)
            print(f"Epoch {epoch}, FID Score: {fid_score}")
            logging.info(f"Epoch {epoch}, FID Score: {fid_score}")
            ms_ssim_score = calculate_MS_SSIM(netG, testloader, gx_test, device)
            print(f"Epoch {epoch}, MS-SSIM Score: {ms_ssim_score}")
            logging.info(f"Epoch {epoch}, MS-SSIM Score: {ms_ssim_score}")

            if ms_ssim_score <= best_ms_ssim_score and fid_score <= best_fid_score:
                print('Save model with best MS-SSIM and FID scores at epoch', epoch)
                logging.info('Save model with best MS-SSIM and FID scores at epoch', epoch)
                best_ms_ssim_score = ms_ssim_score
                best_fid_score = fid_score
                no_improvement_count = 0

                if not os.path.exists('result/model'):
                    os.makedirs('result/model')
                best_model_path = "result/model/best_model_ms_ssim_fid.pth"
                torch.save(netG.state_dict(), best_model_path)
            else:
                no_improvement_count += 1

        # create a fixed noise vector for visualization of the generator output
        image_fixed = [f'data/image/spots_test/test/{i}.jpg' for i in random_indices]
        image_tensors = []
        for img_path in image_fixed:
            img = Image.open(img_path)
            img_tensor = ToTensor()(img)  # Convert to tensor
            image_tensors.append(img_tensor)
        real_images = torch.stack(image_tensors)

        gene_expression_fixed = np.array([gx_test[i] for i in random_indices])
        tensor = torch.tensor(gene_expression_fixed, dtype=torch.float32)
        tensor = tensor.unsqueeze(2).unsqueeze(3)
        fixed_noise = tensor.to(device)

        random_noise = torch.randn(num_predict_image, nz, 1, 1, device=device)
        min_noise, max_noise = random_noise.min(), random_noise.max()
        scaled_noise = 2 * (random_noise - min_noise) / (max_noise - min_noise) - 1
        scaled_noise = scaled_noise.to(device)
        fixed_noise = fixed_noise + noise_scale * scaled_noise

        # train
        train(netG, netD, device, trainloader, optimizerG, optimizerD, criterion, epoch, iters, noise_scale,
              real_images, fixed_noise)

        if epoch % 50 == 0 and epoch >= 300:
            if not os.path.exists('result/model'):
                os.makedirs('result/model')
            torch.save(netG.state_dict(), f"result/model/model_epoch_{epoch}.h5")

    data_loss = []
    for loss_d, loss_g, d_r, d_f1, d_f2, loss_g1, loss_g2 in zip(losses_d, losses_g, d_real, d_fake1, d_fake2, losses_g_bce, losses_g_fm):
        data_loss.append((loss_d, loss_g, d_r, d_f1, d_f2, loss_g1, loss_g2))
    data_loss = pd.DataFrame(data_loss, columns=['loss_d', 'loss_g', 'd_real', 'd_fake1', 'd_fake2', 'loss_g_bce', 'loss_g_fm'])
    data_loss.to_csv('result/model/losses.csv')


def predict_validate(checkpoint_path):
    netG = Generator(ngpu)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    netG.to(device)

    netG.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    gene_expression = Utils.load_gene_expression(gene_expression_validation , noise_scale)

    for i, gene in enumerate(gene_expression):
        input_tensor = torch.tensor(gene, dtype=torch.float32).view(1, nz, 1, 1).to(device)
        random_noise = torch.randn(len(gene_expression), nz, 1, 1, device=device)
        min_noise, max_noise = random_noise.min(), random_noise.max()
        scaled_noise = 2 * (random_noise - min_noise) / (max_noise - min_noise) - 1
        scaled_noise = scaled_noise.to(device)
        input_tensor = input_tensor + noise_scale * scaled_noise
        output_image = netG(input_tensor)
        output_image = output_image.to('cpu').detach()
        output_image = (output_image + 1) / 2.0
        output_image_np = output_image.squeeze(0).permute(1, 2, 0).numpy()
        plt.imsave(f'result/validation/{i}.jpg', output_image_np)
