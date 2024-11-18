from wandb.integration.torch.wandb_torch import torch

from gan import wgan_div, wgan_gp
from gan.model_config import gene_expression_validation, noise_scale
from gan.utils import Utils
from gan.wgan_div import predict_image
from gan.dcgan import predict_validate
from gan.wgan_gp import predict_and_concat

if __name__ == '__main__':
    # wgan_div.run()
    gene_expression = Utils.load_gene_expression(gene_expression_validation, noise_scale)
    # wgan_gp.predict_image('/home/anlt69/Downloads/Wgan_13Nov2024/Wgan_13Nov2024_1/model/model_epoch_900.h5', gene_expression)


    # Example usage
    models = [
        '/home/anlt69/Downloads/Wgan_13Nov2024/Wgan_13Nov2024_1/model/model_epoch_400.h5',
        '/home/anlt69/Downloads/Wgan_13Nov2024/Wgan_13Nov2024_1/model/model_epoch_450.h5',
        '/home/anlt69/Downloads/Wgan_13Nov2024/Wgan_13Nov2024_1/model/model_epoch_500.h5',
        '/home/anlt69/Downloads/Wgan_13Nov2024/Wgan_13Nov2024_1/model/model_epoch_550.h5',
        '/home/anlt69/Downloads/Wgan_13Nov2024/Wgan_13Nov2024_1/model/model_epoch_600.h5',
        '/home/anlt69/Downloads/Wgan_13Nov2024/Wgan_13Nov2024_1/model/model_epoch_650.h5',
        '/home/anlt69/Downloads/Wgan_13Nov2024/Wgan_13Nov2024_1/model/model_epoch_750.h5',
        '/home/anlt69/Downloads/Wgan_13Nov2024/Wgan_13Nov2024_1/model/model_epoch_800.h5',
        '/home/anlt69/Downloads/Wgan_13Nov2024/Wgan_13Nov2024_1/model/model_epoch_850.h5',
        '/home/anlt69/Downloads/Wgan_13Nov2024/Wgan_13Nov2024_1/model/model_epoch_900.h5',
        '/home/anlt69/Downloads/Wgan_13Nov2024/Wgan_13Nov2024_1/model/model_epoch_950.h5',

    ]
    # models = [
    #     '/home/anlt69/Downloads/WganDiv_13Nov2024/model/model_epoch_400.h5',
    #     '/home/anlt69/Downloads/WganDiv_13Nov2024/model/model_epoch_450.h5',
    #     '/home/anlt69/Downloads/WganDiv_13Nov2024/model/model_epoch_500.h5',
    #     '/home/anlt69/Downloads/WganDiv_13Nov2024/model/model_epoch_550.h5',
    #     '/home/anlt69/Downloads/WganDiv_13Nov2024/model/model_epoch_600.h5',
    #     '/home/anlt69/Downloads/WganDiv_13Nov2024/model/model_epoch_650.h5',
    #     '/home/anlt69/Downloads/WganDiv_13Nov2024/model/model_epoch_750.h5',
    #     '/home/anlt69/Downloads/WganDiv_13Nov2024/model/model_epoch_800.h5',
    #     '/home/anlt69/Downloads/WganDiv_13Nov2024/model/model_epoch_850.h5',
    #     '/home/anlt69/Downloads/WganDiv_13Nov2024/model/model_epoch_900.h5',
    #     '/home/anlt69/Downloads/WganDiv_13Nov2024/model/model_epoch_950.h5',
    #
    # ]
    i = 94
    gx =  gene_expression[i]  # Example gene expression data
    real_image_path = f'/home/anlt69/Downloads/GAN_model_13Nov2024/gan_model/data/image_1/spots_validation/validation/{i}.jpg'
    device = 'cpu'
    noise_scale = 0.2
    nz = 2048  # Example noise dimension

    predict_and_concat(models, gx, real_image_path, device, noise_scale, nz)

    wgan_gp.predict_image('/home/anlt69/Downloads/WganDiv_13Nov2024/model/model_epoch_650.h5', gene_expression)