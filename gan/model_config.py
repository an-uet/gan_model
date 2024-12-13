# Parameters

log_interval = 400

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 1024

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 2000

# Learning rate for optimizers
lr_d = 0.0002
lr_g = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

num_predict_image = 32

no_cuda = False

real_label = 0.9

fake_label = 0.0

noise_scale = 0.3

# additional parameters for wgan
n_critic = 5

alpha = 0.3

clip_value = 0.01

# additional parameters for wgan-gp
lambda_gp = 10

dataroot_train = 'data/image/spots_train'
dataroot_test = 'data/image/spots_test'
gene_expression_data = 'data/gene_expression/top_1024/data_train.csv'
gene_expression_test = 'data/gene_expression/top_1024/data_test.csv'
gene_expression_validation = 'data/gene_expression/top_1024/data_validation.csv'
random_indices = [483, 38, 355, 494, 393, 108, 193, 209, 358, 213, 280, 380, 162, 306, 347, 398, 110, 123, 141, 156,
                  324, 1, 59, 51, 130, 131, 5, 299, 319, 352, 64, 256]
