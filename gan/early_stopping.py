# Early Stopping Criteria for Training Generative Adversarial Networks in Biomedical Imaging
# https://arxiv.org/html/2405.20987v1

import torch
import numpy as np
from pytorch_msssim import ms_ssim
from scipy.linalg import sqrtm
from gan.model_config import batch_size


def early_stopping(train_loader, test_loader, generator, discriminator, max_epochs, MS_SSIM_Th1, MS_SSIM_Th2, FID_Th1,
                   FID_Th2, patience, criterion, optimizerG, optimizerD, device):
    best_MS_SSIM_score = min(MS_SSIM_Th1, MS_SSIM_Th2)
    best_FID_score = min(FID_Th1, FID_Th2)
    no_improvement_count = 0
    loss_problem_count = 0
    recent_loss_G = []
    recent_loss_D = []

    for epoch in range(1, max_epochs + 1):
        # Train for one epoch
        train_loss_G, train_loss_D = train_one_epoch(generator, discriminator, train_loader, criterion, optimizerG,
                                                     optimizerD, device)

        # Record the loss values
        recent_loss_G.append(train_loss_G)
        recent_loss_D.append(train_loss_D)

        # Analyze loss patterns
        if analyze_loss_patterns(recent_loss_G, recent_loss_D):
            loss_problem_count += 1
        else:
            loss_problem_count = 0

        # Early stopping based on loss patterns
        if loss_problem_count >= patience:
            print("Early stopping due to training problems detected in loss values.")
            break

        # Every 50 epochs, check MS-SSIM and FID scores
        if epoch % 50 == 0:
            if constant_loss_values(recent_loss_G, recent_loss_D):
                MS_SSIM_score = calculate_MS_SSIM(generator, test_loader, gene_expression, device)
                FID_score = calculate_FID(generator, train_loader, test_loader, device)

                if MS_SSIM_score <= best_MS_SSIM_score and FID_score <= best_FID_score:
                    best_MS_SSIM_score = MS_SSIM_score
                    best_FID_score = FID_score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

        # Early stopping based on no improvement in MS-SSIM and FID scores
        if no_improvement_count >= patience:
            print("Early stopping due to no improvement in MS-SSIM and FID scores.")
            break

    return recent_loss_G, recent_loss_D, best_MS_SSIM_score, best_FID_score


def train_one_epoch(generator, discriminator, dataloader, criterion, optimizerG, optimizerD, device):
    # Implement the training loop for one epoch
    # Return the generator and discriminator loss for the epoch
    pass


def analyze_loss_patterns(recent_loss_G, recent_loss_D):
    # Implement the logic to analyze loss patterns
    # Return True if problems are detected, otherwise False
    pass


def constant_loss_values(recent_loss_G, recent_loss_D):
    # Implement the logic to check for constant loss values
    # Return True if constant loss values are detected, otherwise False
    pass


def calculate_MS_SSIM(generator, dataloader, gene_expression, device):
    ms_ssim_score = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            gene_expression_batch = gene_expression[i * batch_size:(i + 1) * batch_size]
            tensor_gene = torch.tensor(gene_expression_batch, dtype=torch.float32)
            tensor_gene = tensor_gene.unsqueeze(2).unsqueeze(3)
            noise = tensor_gene.to(device)
            fake_images = generator(noise)

            ms_ssim_score += ms_ssim(real_images, fake_images, data_range=1.0).item()
            num_batches += 1
    return ms_ssim_score / num_batches


def calculate_FID(generator, dataloader, encoder, gene_expression, device):
    def get_features(loader):
        features = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                encoded_features = encoder(images)
                features.append(encoded_features.cpu().numpy())
        return np.concatenate(features, axis=0)

    # Get features for real images
    real_features = get_features(dataloader)

    # Generate fake images and get their features
    fake_features = []
    with torch.no_grad():
        for i in range(len(dataloader)):
            gene_expression_batch = gene_expression[i * batch_size:(i + 1) * batch_size]
            tensor_gene = torch.tensor(gene_expression_batch, dtype=torch.float32)
            tensor_gene = tensor_gene.unsqueeze(2).unsqueeze(3)
            noise = tensor_gene.to(device)
            fake_images = generator(noise)
            encoded_features = encoder(fake_images)
            fake_features.append(encoded_features.cpu().numpy())
    fake_features = np.concatenate(fake_features, axis=0)

    # Calculate mean and covariance of real and fake features
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    # Calculate FID score
    diff = mu_real - mu_fake
    covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid_score
