# backup encoder model

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class Encoder(nn.Module):
    def __init__(self, input_channels=3, encoding_dim=2048):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),  # Dropout to prevent overfitting

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),  # Dropout to prevent overfitting

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),  # Dropout to prevent overfitting

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)  # Dropout to prevent overfitting
        )
        self.fc = nn.Linear(16 * 16 * 512, encoding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoding_dim=2048, output_channels=3):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(encoding_dim, 16 * 16 * 512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 16, 16)
        x = self.decoder(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_channels=3, encoding_dim=2048):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, encoding_dim=encoding_dim)
        self.decoder = Decoder(encoding_dim=encoding_dim, output_channels=input_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def concat_image(image1, image2, dest):
    # Ensure both images have the same height for horizontal concatenation
    # You can also resize them to match, if needed
    if image1.size[1] != image2.size[1]:
        new_height = min(image1.size[1], image2.size[1])
        image1 = image1.resize((int(image1.size[0] * new_height / image1.size[1]), new_height))
        image2 = image2.resize((int(image2.size[0] * new_height / image2.size[1]), new_height))

    # Calculate the size of the new image
    total_width = image1.size[0] + image2.size[0]
    max_height = max(image1.size[1], image2.size[1])

    # Create a new image with the combined size
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste the images into the new image
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.size[0], 0))

    # Save the concatenated image
    output_path = dest
    new_image.save(output_path)


# def encode_image(image, encoder, device):
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         encoding_vector = encoder(image_tensor)
#     return encoding_vector


from torchvision import transforms
import torch


def encode_image(image, encoder, device):
    # Check if image is a tensor
    if isinstance(image, torch.Tensor):
        # If image is already a tensor, no need to apply transforms
        image_tensor = image
    else:
        # Apply transforms if image is a PIL Image or ndarray
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        image_tensor = transform(image)

    # Add a batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Compute encoding vector
    with torch.no_grad():
        encoding_vector = encoder(image_tensor)

    return encoding_vector


def custom_loss(encoder, device, fake_images, real_images, disc_output, criterion, sim_loss_weight=0.5, epsilon=1,
                label=1):
    # Calculate image similarity loss
    imgSim = []
    for i in range(len(fake_images)):
        # Convert data to grayscale to get intensity value
        vector1 = encode_image(fake_images[i], encoder, device)
        vector2 = encode_image(real_images[i], encoder, device)
        # Compute Euclidean distance between them
        simVal = np.linalg.norm(vector2 - vector1)
        imgSim.append(simVal)
    imgSim = torch.tensor(imgSim, device=fake_images.device, dtype=torch.float32)

    # Apply logarithmic scale
    imgSim_log = torch.log(imgSim + epsilon)  # Adding epsilon to avoid log(0)

    # Calculate mean of log-scaled imgSim
    imgSim_loss = sim_loss_weight * imgSim_log.mean()  # Optionally scale the similarity loss

    # Calculate BCELoss
    bce_loss = (1 - sim_loss_weight) * criterion(disc_output, torch.full_like(disc_output, label))  # Target is real (1)

    # Combine both losses
    combined_loss = bce_loss + imgSim_loss
    return combined_loss


def custom_loss_mse(encoder, device, fake_images, real_images, disc_output, criterion, sim_loss_weight=0.5, label=1):
    # Calculate image similarity loss
    imgSim = []
    for i in range(len(fake_images)):
        # Get encoded vectors for fake and real images
        vector1 = encode_image(fake_images[i], encoder, device)
        vector2 = encode_image(real_images[i], encoder, device)

        # Compute MSE between the two vectors
        simVal = F.mse_loss(vector1, vector2, reduction='none').mean()
        imgSim.append(simVal)

    imgSim = torch.tensor(imgSim, device=fake_images.device, dtype=torch.float32)

    # Calculate the mean of the similarity loss
    imgSim_loss = sim_loss_weight * imgSim.mean()  # Optionally scale the similarity loss

    # Calculate BCELoss for the discriminator output
    bce_loss = (1 - sim_loss_weight) * criterion(disc_output, torch.full_like(disc_output, label))  # Target is real (1)

    # Combine both losses
    combined_loss = bce_loss + imgSim_loss
    return combined_loss


if __name__ == '__main__':
    # Load the saved autoencoder model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = Autoencoder(input_channels=3, encoding_dim=2048)
    autoencoder.load_state_dict(
        torch.load('/data/haunt/st2image/image_encoder/encoder_model/best_model_mse_ssim_2048_0.1.pth',
                   map_location=device))
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    # Separate the encoder and decoder
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    # Load the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open('/data/haunt/st2image/data/images/train_gan/all_images/B4_sub_img_11_17.jpg')
    print(encode_image(image, encoder, device))

    # Get the encoding vector from the encoder
    #     with torch.no_grad():
    #         encoding_vector = encoder(image_tensor)

    #     # Print or save the encoding vector if needed
    #     print(encoding_vector, encoding_vector.shape)

    #     # Reconstruct the image using the decoder
    #     with torch.no_grad():
    #         reconstructed_image = decoder(encoding_vector)
    #         reconstructed_image = reconstructed_image.squeeze(0).cpu()

    #         # Denormalize the image
    #         reconstructed_image = (reconstructed_image * 0.5) + 0.5

    #         # Convert the tensor back to a PIL image
    #         reconstructed_image_pil = transforms.ToPILImage()(reconstructed_image)

    #         # Display and save the reconstructed image
    #         # reconstructed_image_pil.show()
    #         reconstructed_image_pil.save(f'result/reconstructed_image_from_vector_{i}.jpg')

    #         # Concatenate the original and reconstructed images
    #         real_image = Image.open(f'/home/anlt69/Desktop/study/st2image/data_sampled/image_sampled/{i}.jpg')
    #         concat_image(real_image, reconstructed_image_pil, f'reconstructed_image/concatenated_image_{i}.jpg')

    # except Exception as e:
    #     print(e)
