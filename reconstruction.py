import numpy as np
from ksvdvae.models import DeepKSVDVAE
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from scipy import linalg
from img_datasets import create_dataset
import argparse
from ksvdvae.utils.setup import setup
from img_datasets.miniimagenet import MiniImagenet
from results.CompVQs.rqvae.models import create_model
from results.CompVQs.rqvae.metrics.fid import compute_rfid
from results.CompVQs.rqvae.utils.config import load_config, augment_arch_defaults
from results.CompVQs.fid import compute_rfid
import yaml
import torch
from omegaconf import OmegaConf
from results.CompVQs.taming.models.vqgan import VQModel, GumbelVQ
import sys
import piq
sys.path.append(".")
sys.path.append("./results/CompVQs/")
'''
def inspect_model(models):
    param_count = 0
    for param_tensor_str in models['state_dict']:
        tensor_size = models['state_dict'][param_tensor_str].size()
        print(f"{param_tensor_str} size {tensor_size} = {models['state_dict'][param_tensor_str].numel()} params")
        param_count += models['state_dict'][param_tensor_str].numel()

    print(f"Number of parameters: {param_count}")
'''

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _, _ = model(images)
    return x_tilde

def main(args, model_config):
    # load dataset
    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        if args.dataset == 'mnist':
            # Define the train & test datasets
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                                           download=True, transform=transform)
            test_dataset = datasets.MNIST(args.data_folder, train=False, download=True,
                                          transform=transform)
            num_channels = 1
        elif args.dataset == 'fashion-mnist':
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(args.data_folder,
                                                  train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(args.data_folder,
                                                 train=False, transform=transform)
            num_channels = 1
        elif args.dataset == 'cifar10':
            # Define the train & test datasets
            train_dataset = datasets.CIFAR10(args.data_folder,
                                             train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(args.data_folder,
                                            train=False, transform=transform)
            num_channels = 3
        valid_dataset = test_dataset
    elif args.dataset == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(args.data_folder, train=True, download=True, transform=transform)
        valid_dataset = MiniImagenet(args.data_folder, valid=True, download=True, transform=transform)
        test_dataset = MiniImagenet(args.data_folder, test=True, download=True, transform=transform)
        num_channels = 3
    elif args.dataset == 'FFHQ':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'imagenet':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'LSUN-cat':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'LSUN-church':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'LSUN-bedroom':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'cc3m':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'caltech101':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'caltech256':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    else:
        raise NotImplementedError('%s not implemented..' % args.dataset)


    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=4, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=4, shuffle=False, drop_last=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=8, shuffle=False)

    Hidden_size, H_1, H_2, num_soft_thresh = model_config.arch.vae.hidden_size, model_config.arch.alpha.H_1, model_config.arch.alpha.H_2, model_config.arch.latent.num_soft_thresh

    # Initialization of num of atoms
    Dict_init = DeepKSVDVAE.init_dct(int(np.sqrt(Hidden_size)), 23)
    num_atoms = min(23 * 23, model_config.arch.latent.num_atoms)
    Dict_init = Dict_init[:, :num_atoms].to(args.device)
    #Dict_init2 = Dict_init.detach().clone()
    #visualize_dictionary_as_atoms(Dict_init)

    #Dict_init = np.random.uniform(low=-1, high=1, size=(256, 512))
    #Dict_init = torch.from_numpy(Dict_init).float().to(args.device)

    # The matrix 2-nrom
    c_init = (linalg.norm(Dict_init.cpu(), ord=2)) ** 2
    c_init = torch.FloatTensor((c_init,))
    c_init = c_init.to(args.device)

    w_init = torch.normal(mean=1, std=1 / 10 * torch.ones(Hidden_size)).float()
    w_init = w_init.to(args.device)

    '''
    model = DeepKSVDVAE.Model_VAEf4(
        num_channels,
        Hidden_size,
        H_1,
        H_2,
        model_config.arch.latent.num_soft_thresh,
        Dict_init,
        c_init,
        w_init,
        model_config.arch.latent.beta,
        args.device,
    )
    '''

    '''
    model = DeepKSVDVAE.Model_VAEf16(
        model_config.arch.vae.ddconfig,
        num_channels,
        Hidden_size,
        H_1,
        H_2,
        model_config.arch.latent.num_soft_thresh,
        Dict_init,
        c_init,
        w_init,
        model_config.arch.latent.beta,
        args.device,
    )
    model.to(args.device)
    '''

    #FFHQ
    #ModelPath = './models/deepksvd-vae_learnable_alpha_beta_FFHQ_num-epochs20_T1_2022_09_19_07_57_11/best.pt'
    #ModelPath = './models/deepksvd-vae_learnable_alpha_beta_FFHQ_num-epochs20_T5_2022_09_19_00_31_52/best.pt'
    #ModelPath = './models/deepksvd-vae_learnable_alpha_beta_FFHQ_num-epochs20_T8_2022_09_19_07_59_30/best.pt'
    #ModelPath = './models/FFHQ/deepksvd-vae_learnable_alpha_beta_FFHQ_num-epochs20_T5_beta5_2022_09_19_16_21_10/best.pt'
    #ModelPath = './models/FFHQ/deepksvd-vae_learnable_alpha_beta_FFHQ_num-epochs20_T5_beta10_2022_09_19_16_25_44/best.pt'

    #fashion-mnist
    #ModelPath = './models/fashion-mnist/deepksvd-vae_learnable_alpha_beta_fashion-mnist_num-epochs100_T5_beta1_2022_09_19_23_22_28/best.pt'
    #ModelPath = './models/fashion-mnist/deepksvd-vae_learnable_alpha_beta_fashion-mnist_num-epochs100_T5_beta3_2022_09_19_17_21_47/best.pt'
    #ModelPath = './models/fashion-mnist/deepksvd-vae_learnable_alpha_beta_fashion-mnist_num-epochs100_T5_beta5_2022_09_19_23_24_58/best.pt'

    #LSUN-church
    #ModelPath = './results/models/LSUN-church/deepksvd-vae_learnable_alpha_fixed_beta_LSUN-church_num-epochs50_num_soft_thresh5_beta5_num_atoms512_2022_09_25_00_44_49/best.pt'

    #FFHQ-DCT
    #ModelPath_beta1 = './results/models/FFHQ/deepksvd-vae_learnable_alpha_fixed_beta_FFHQ_num-epochs20_batchsize16_num_soft_thresh5_beta1_num_atoms512_2022_09_30_11_04_31_ffhq256-16x16/best.pt'
    #ModelPath_beta2 = './results/models/FFHQ/deepksvd-vae_learnable_alpha_fixed_beta_FFHQ_num-epochs20_batchsize16_num_soft_thresh5_beta2_num_atoms512_2022_09_29_12_29_55_ffhq256-16x16/best.pt'
    #ModelPath_beta3 = './results/models/FFHQ/deepksvd-vae_learnable_alpha_fixed_beta_FFHQ_num-epochs50_batchsize16_num_soft_thresh5_beta3_num_atoms512_2022_10_03_10_13_08_ffhq256-16x16/best.pt'
    #ModelPath_beta4 = './results/models/FFHQ/deepksvd-vae_learnable_alpha_fixed_beta_FFHQ_num-epochs50_batchsize16_num_soft_thresh5_beta4_num_atoms512_2022_10_04_10_38_49_ffhq256-16x16/best.pt'
    #ModelPath_beta5 = './results/models/FFHQ/deepksvd-vae_learnable_alpha_fixed_beta_FFHQ_num-epochs20_batchsize16_num_soft_thresh5_beta5_num_atoms512_2022_09_29_12_36_53_ffhq256-16x16/best.pt'


    #FFHQ-random
    #ModelPath = './results/models/FFHQ/random_FFHQ_num-epochs50_batchsize16_num_soft_thresh5_beta20_num_atoms512_2022_10_08_23_43_09_ffhq256-deepksvdvae16x16/best.pt'
    #ModelPath = './results/models/FFHQ/random_FFHQ_num-epochs50_batchsize16_num_soft_thresh5_beta10_num_atoms512_2022_10_08_23_19_16_ffhq256-deepksvdvae16x16/best.pt'
    #ModelPath = './results/models/FFHQ/random_FFHQ_num-epochs50_batchsize16_num_soft_thresh5_beta5_num_atoms512_2022_10_08_23_16_11_ffhq256-deepksvdvae16x16/best.pt'
    #ModelPath = './results/models/FFHQ/random_FFHQ_num-epochs50_batchsize16_num_soft_thresh5_beta1_num_atoms512_2022_10_08_23_15_03_ffhq256-deepksvdvae16x16/best.pt'

    #cifar10
    #ModelPath = './results/models/cifar10/deepksvd-vae_learnable_alpha_fixed_beta_cifar10_num-epochs50_batchsize32_num_soft_thresh5_beta1_num_atoms512_2022_10_01_17_08_49_cifar-deepksvdvae2x2/best.pt'

    #caltech101
    #ModelPath = './results/models/caltech101/deepksvd-vae_learnable_alpha_fixed_beta_caltech101_num-epochs50_batchsize8_num_soft_thresh5_beta1_num_atoms512_2022_10_03_16_10_38_caltech101_16x16/best.pt'
    #ModelPath = './results/models/caltech101/deepksvd-vae_learnable_alpha_fixed_beta_caltech101_num-epochs50_batchsize8_num_soft_thresh5_beta1_num_atoms512_2022_10_04_11_13_21_caltech101_16x16/best.pt'
    #ModelPath = './results/models/caltech101/deepksvd-vae_learnable_alpha_fixed_beta_caltech101_num-epochs50_batchsize8_num_soft_thresh5_beta5_num_atoms512_2022_10_04_11_14_30_caltech101_16x16/best.pt'

    #Imagenet
    #ModelPath_beta2 = './results/models/imagenet/multipleGPUs_imagenet_num-epochs50_batchsize16_num_soft_thresh5_beta2_num_atoms512_2022_10_15_13_27_29_im256-deepksvdvae16x16/best.pt'
    #ModelPath_beta3 = './results/models/imagenet/multipleGPUs_imagenet_num-epochs50_batchsize12_num_soft_thresh5_beta3_num_atoms512_2022_10_16_02_23_41_im256-deepksvdvae16x16/best.pt'


    '''
    #imagenet 32x32
    # --model-config ./configs/imagenet256/stage1/im256-deepksvdvae32x32.yaml
    ModelPath_beta2 = './results/models/imagenet/multipleGPUs_imagenet_num-epochs50_batchsize12_num_soft_thresh5_beta2_num_atoms512_2022_10_17_20_13_06_im256-deepksvdvae32x32/best.pt'
    #args.model_config = './configs/imagenet256/stage1/im256-deepksvdvae32x32.yaml'
    model_config.arch.latent.beta = 2
    deepksvdvae_model_beta2 = DeepKSVDVAE.Model_VAEf16(model_config.arch.vae.ddconfig, num_channels, Hidden_size,
                                                       H_1, H_2, model_config.arch.latent.num_soft_thresh, Dict_init,
                                                       c_init, w_init, model_config.arch.latent.beta, args.device)
    deepksvdvae_model_beta2.to(args.device)
    deepksvdvae_model_beta2.load_state_dict(torch.load(ModelPath_beta2, map_location=args.device))
    deepksvdvae_model_beta2.eval()
    '''


    '''
    # imagenet 16x16
    # --model-config ./configs/imagenet256/stage1/im256-deepksvdvae16x16.yaml
    ModelPath_beta2 = './results/models/imagenet/multipleGPUs_imagenet_num-epochs50_batchsize16_num_soft_thresh5_beta2_num_atoms512_2022_10_15_13_27_29_im256-deepksvdvae16x16/best.pt'
    model_config.arch.latent.beta = 2
    deepksvdvae_model_beta2 = DeepKSVDVAE.Model_VAEf16(model_config.arch.vae.ddconfig, num_channels, Hidden_size,
                                                       H_1, H_2, model_config.arch.latent.num_soft_thresh, Dict_init,
                                                       c_init, w_init, model_config.arch.latent.beta, args.device)
    deepksvdvae_model_beta2.to(args.device)
    deepksvdvae_model_beta2.load_state_dict(torch.load(ModelPath_beta2, map_location=args.device))
    deepksvdvae_model_beta2.eval()
    '''



    '''
    # imagenet 16x16
    ModelPath_beta3 = './results/models/imagenet/multipleGPUs_imagenet_num-epochs50_batchsize12_num_soft_thresh5_beta3_num_atoms512_2022_10_16_02_23_41_im256-deepksvdvae16x16/best.pt'
    model_config.arch.latent.beta = 3
    deepksvdvae_model_beta3 = DeepKSVDVAE.Model_VAEf16(model_config.arch.vae.ddconfig, num_channels, Hidden_size,
                                                       H_1, H_2, model_config.arch.latent.num_soft_thresh, Dict_init,
                                                       c_init, w_init, model_config.arch.latent.beta, args.device)
    deepksvdvae_model_beta3.to(args.device)
    deepksvdvae_model_beta3.load_state_dict(torch.load(ModelPath_beta3, map_location=args.device))
    deepksvdvae_model_beta3.eval()
    '''



    #config1024 = load_config("results/CompVQs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
    #model1024 = load_vqgan(config1024, ckpt_path="results/CompVQs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(
    #    args.device)
    #config16384 = load_config("results/CompVQs/vqgan_imagenet_f16_16384/configs/model.yaml", display=False)
    #model16384 = load_vqgan(config16384, ckpt_path="results/CompVQs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(
    #    args.device)

    #config32x32 = load_config("results/CompVQs/vqgan_gumbel_f8/configs/model.yaml", display=False)
    #model32x32 = load_vqgan(config32x32, ckpt_path="results/CompVQs/vqgan_gumbel_f8/checkpoints/last.ckpt",
    #                        is_gumbel=True).to(args.device)


    '''
    #DALLE
    from results.CompVQs.dall_e import load_model
    encoder_dalle = load_model("https://cdn.openai.com/dall-e/encoder.pkl", args.device)
    decoder_dalle = load_model("https://cdn.openai.com/dall-e/decoder.pkl", args.device)
    dalle_model = [encoder_dalle, decoder_dalle]
    '''



    '''
    #RQVAE-FFHQ
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Batch size to use')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--vqvae', type=str, default='./results/CompVQs/rqvae_ffhq_pre-trained/stage1/model.pt',
                        help='vqvae path for recon FID')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rqvae_model, config = load_rqvae_model(args.vqvae)
    rqvae_model = rqvae_model.to(args.device)
    '''

    
    #RQVAE-Imagenet
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=12,
                            help='Batch size to use')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--vqvae', type=str, default='./results/CompVQs/rqvae_imagenet_821M/stage1/model.pt',
                            help='vqvae path for recon FID')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rqvae_model, config = load_rqvae_model(args.vqvae)
    rqvae_model = rqvae_model.to(args.device)
    




    '''
    ####beta1
    model_config.arch.latent.beta = 1
    deepksvdvae_model_beta1 = DeepKSVDVAE.Model_VAEf16(model_config.arch.vae.ddconfig, num_channels, Hidden_size,
                                                       H_1, H_2, model_config.arch.latent.num_soft_thresh, Dict_init,
                                                       c_init, w_init, model_config.arch.latent.beta, args.device)
    deepksvdvae_model_beta1.to(args.device)
    deepksvdvae_model_beta1.load_state_dict(torch.load(ModelPath_beta1, map_location=args.device));
    deepksvdvae_model_beta1.eval()
    '''

    '''
    ####beta2
    model_config.arch.latent.beta = 2
    deepksvdvae_model_beta2 = DeepKSVDVAE.Model_VAEf16(model_config.arch.vae.ddconfig, num_channels, Hidden_size,
                                                       H_1, H_2, model_config.arch.latent.num_soft_thresh, Dict_init,
                                                       c_init, w_init, model_config.arch.latent.beta, args.device)
    deepksvdvae_model_beta2.to(args.device)
    deepksvdvae_model_beta2.load_state_dict(torch.load(ModelPath_beta2, map_location=args.device));
    deepksvdvae_model_beta2.eval()
    '''


    '''
    ####beta3
    model_config.arch.latent.beta = 3
    deepksvdvae_model_beta3 = DeepKSVDVAE.Model_VAEf16(model_config.arch.vae.ddconfig, num_channels, Hidden_size,
                                                       H_1, H_2, model_config.arch.latent.num_soft_thresh, Dict_init,
                                                       c_init, w_init, model_config.arch.latent.beta, args.device)
    deepksvdvae_model_beta3.to(args.device)
    deepksvdvae_model_beta3.load_state_dict(torch.load(ModelPath_beta3, map_location=args.device));
    deepksvdvae_model_beta3.eval()
    deepksvdvae_model_beta3.load_state_dict(torch.load(ModelPath_beta3, map_location=args.device));
    deepksvdvae_model_beta3.eval()
    '''

    '''
    ####beta4
    model_config.arch.latent.beta = 4
    deepksvdvae_model_beta4 = DeepKSVDVAE.Model_VAEf16(model_config.arch.vae.ddconfig, num_channels, Hidden_size,
                                                       H_1, H_2, model_config.arch.latent.num_soft_thresh, Dict_init,
                                                       c_init, w_init, model_config.arch.latent.beta, args.device)
    deepksvdvae_model_beta4.to(args.device)
    deepksvdvae_model_beta4.load_state_dict(torch.load(ModelPath_beta4, map_location=args.device));
    deepksvdvae_model_beta4.eval()
    '''


    '''
    ####beta5
    model_config.arch.latent.beta = 5
    deepksvdvae_model_beta5 = DeepKSVDVAE.Model_VAEf16(model_config.arch.vae.ddconfig, num_channels, Hidden_size,
                                                       H_1, H_2, model_config.arch.latent.num_soft_thresh, Dict_init,
                                                       c_init, w_init, model_config.arch.latent.beta, args.device)
    deepksvdvae_model_beta5.to(args.device)
    deepksvdvae_model_beta5.load_state_dict(torch.load(ModelPath_beta5, map_location=args.device));
    deepksvdvae_model_beta5.eval()
    '''

    '''
    # compute rfid for VQGAN
    print('32x32:')
    rfid_vqgan_32x32 = compute_rfid(test_dataset, model32x32, batch_size=320, device=args.device)
    print(rfid_vqgan_32x32)
    print('1024:')
    rfid_vqgan_1024 = compute_rfid(test_dataset, model1024, batch_size=320, device=args.device)
    print(rfid_vqgan_1024)
    print('16384:')
    rfid_vqgan_16384 = compute_rfid(test_dataset, model16384, batch_size=320, device=args.device)
    print(rfid_vqgan_16384)
    '''

    # compute rfid for DALLE
    #rfid_dalle = compute_rfid(test_dataset, dalle_model, batch_size=32, device=args.device)
    #print(rfid_dalle)

    # compute rfid for rqvae ffhq
    rfid_rqvae = compute_rfid(test_dataset, rqvae_model, batch_size=32, device=args.device)
    print(rfid_rqvae, flush=True)

    # compute rfid for rqvae imagenet
    #rfid_rqvae = compute_rfid(test_dataset, rqvae_model, batch_size=32, device=args.device)
    #print(rfid_rqvae)


    # compute rfid for rqvae imagenet
    #rfid_rqvae = compute_rfid(test_dataset, rqvae_model, batch_size=32, device=args.device)
    #print(rfid_rqvae)

    #KSVDVAE FFHQ
    #rfid = compute_rfid(test_dataset, deepksvdvae_model_beta4, batch_size=16, device=args.device)
    #print(rfid)

    # KSVDVAE Imagenet 32x32
    #rfid = compute_rfid(test_dataset, deepksvdvae_model_beta2, batch_size=16, device=args.device)
    #print(rfid)

    # KSVDVAE Imagenet 16x16
    #rfid = compute_rfid(test_dataset, deepksvdvae_model_beta2, batch_size=16, device=args.device)
    #rfid = compute_rfid(test_dataset, deepksvdvae_model_beta3, batch_size=16, device=args.device)
    #print(rfid, flush=True)


    # compute psnr and ssim
    from results.CompVQs.dall_e import map_pixels, unmap_pixels, load_model
    import lpips
    #psnrs_beta1 = []; ssims_beta1 = []
    psnrs_beta2 = []; ssims_beta2 = []; lpips_beta2 = []
    loss_fn_alex = lpips.LPIPS(net='alex').to(args.device)  # best forward scores
    #psnrs_beta3 = []; ssims_beta3 = []
    #psnrs_beta4 = []; ssims_beta4 = []
    #psnrs_beta5 = []; ssims_beta5 = []
    with torch.no_grad():
        for images, target in test_loader:
            #VQGAN
            #images = images.to(args.device)
            #x_tilde_beta2= reconstruct_with_vqgan(images, model16384)
            #images = torch.clamp(images * 0.5 + 0.5, 0, 1)
            #x_tilde_beta2 = torch.clamp(x_tilde_beta2 * 0.5 + 0.5, 0, 1)
            #RQVAE
            images = images.to(args.device)
            x_tilde_beta2, _, _ = rqvae_model(images)
            images = torch.clamp(images * 0.5 + 0.5, 0, 1)
            x_tilde_beta2 = torch.clamp(x_tilde_beta2 * 0.5 + 0.5, 0, 1)
            #DALLE: convert to 0-1 at first
            #images = images.to(args.device)
            #images = torch.clamp(images * 0.5 + 0.5, 0, 1)
            #images = map_pixels(images)
            #x_tilde_beta2 = reconstruct_with_dalle(images, dalle_model[0], dalle_model[1])
            #images = torch.clamp(images * 0.5 + 0.5, 0, 1)
            #x_tilde_beta2 = torch.clamp(x_tilde_beta2 * 0.5 + 0.5, 0, 1)
            # KSVDVAE
            #images = images.to(args.device)
            #x_tilde_beta1, _, _, z1 = deepksvdvae_model_beta1(images)
            #x_tilde_beta2, _, _, z2 = deepksvdvae_model_beta2(images)
            #x_tilde_beta3, _, _, z3 = deepksvdvae_model_beta3(images)
            #x_tilde_beta4, _, _, z4 = deepksvdvae_model_beta4(images)
            #x_tilde_beta5, _, _, z5 = deepksvdvae_model_beta5(images)
            lp_beta2 = loss_fn_alex(images, x_tilde_beta2)
            lp_beta2 = torch.squeeze(lp_beta2); print(lp_beta2, flush=True)
            # we are assuming that dataset returns value in -1 ~ 1 -> remap to 0 ~ 1
            #x_tilde_beta1 = torch.clamp(x_tilde_beta1 * 0.5 + 0.5, 0, 1)
            #x_tilde_beta2 = torch.clamp(x_tilde_beta2 * 0.5 + 0.5, 0, 1)
            #x_tilde_beta3 = torch.clamp(x_tilde_beta3 * 0.5 + 0.5, 0, 1)
            #x_tilde_beta4 = torch.clamp(x_tilde_beta4 * 0.5 + 0.5, 0, 1)
            #x_tilde_beta5 = torch.clamp(x_tilde_beta5 * 0.5 + 0.5, 0, 1)
            #images = torch.clamp(images * 0.5 + 0.5, 0, 1)
            #psnr_beta1 = piq.psnr(images, x_tilde_beta1, data_range=1., reduction='none')
            #ssim_beta1 = piq.ssim(images, x_tilde_beta1, data_range=1., reduction='none')
            psnr_beta2 = piq.psnr(images, x_tilde_beta2, data_range=1., reduction='none')
            ssim_beta2 = piq.ssim(images, x_tilde_beta2, data_range=1., reduction='none')
            #psnr_beta3 = piq.psnr(images, x_tilde_beta3, data_range=1., reduction='none')
            #ssim_beta3 = piq.ssim(images, x_tilde_beta3, data_range=1., reduction='none')
            #psnr_beta4 = piq.psnr(images, x_tilde_beta4, data_range=1., reduction='none')
            #ssim_beta4 = piq.ssim(images, x_tilde_beta4, data_range=1., reduction='none')
            #psnr_beta5 = piq.psnr(images, x_tilde_beta5, data_range=1., reduction='none')
            #ssim_beta5 = piq.ssim(images, x_tilde_beta5, data_range=1., reduction='none')
            #psnrs_beta1.append(psnr_beta1); ssims_beta1.append(ssim_beta1)
            psnrs_beta2.append(psnr_beta2); ssims_beta2.append(ssim_beta2); lpips_beta2.append(lp_beta2)
            #psnrs_beta3.append(psnr_beta3); ssims_beta3.append(ssim_beta3)
            #psnrs_beta4.append(psnr_beta4); ssims_beta4.append(ssim_beta4)
            #psnrs_beta5.append(psnr_beta5); ssims_beta5.append(ssim_beta5)
    #psnr_avg_beta1 = torch.mean(torch.cat(psnrs_beta1)); print('psnr_avg_beta1'+ str(psnr_avg_beta1)); ssim_avg_beta1 = torch.mean(torch.cat(ssims_beta1)); print('ssim_avg_beta1'+ str(ssim_avg_beta1))
    psnr_avg_beta2 = torch.mean(torch.cat(psnrs_beta2)); print('psnr_avg_beta2:'+ str(psnr_avg_beta2)); ssim_avg_beta2 = torch.mean(torch.cat(ssims_beta2)); print('ssim_avg_beta2:'+ str(ssim_avg_beta2)); lpips_avg_beta2 = torch.mean(torch.cat(lpips_beta2)); print('lpips_avg_beta2:'+ str(lpips_avg_beta2))
    #psnr_avg_beta3 = torch.mean(torch.cat(psnrs_beta3)); print('psnr_avg_beta3'+ str(psnr_avg_beta3)); ssim_avg_beta3 = torch.mean(torch.cat(ssims_beta3)); print('ssim_avg_beta3'+ str(ssim_avg_beta3))
    #psnr_avg_beta4 = torch.mean(torch.cat(psnrs_beta4)); print('psnr_avg_beta4'+ str(psnr_avg_beta4)); ssim_avg_beta4 = torch.mean(torch.cat(ssims_beta4)); print('ssim_avg_beta4'+ str(ssim_avg_beta4))
    #psnr_avg_beta5 = torch.mean(torch.cat(psnrs_beta5)); print('psnr_avg_beta5'+ str(psnr_avg_beta5)); ssim_avg_beta5 = torch.mean(torch.cat(ssims_beta5)); print('ssim_avg_beta5'+ str(ssim_avg_beta5))

    print('breakpionts')

from results.CompVQs.dall_e import unmap_pixels
import torch.nn.functional as F
def reconstruct_with_dalle(x, encoder, decoder):

    z_logits = encoder(x)
    z = torch.argmax(z_logits, axis=1)

    print(f"DALL-E: latent shape: {z.shape}")
    z = F.one_hot(z, num_classes=encoder.vocab_size).permute(0, 3, 1, 2).float()

    x_stats = decoder(z).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
    #x_rec = T.ToPILImage(mode='RGB')(x_rec[0])

    return x_rec

def load_rqvae_model(path, ema=False):
    model_config = os.path.join(os.path.dirname(path), 'config.yaml')
    config = load_config(model_config)
    config.arch = augment_arch_defaults(config.arch)

    model, _ = create_model(config.arch, ema=False)
    ckpt = torch.load(path)['state_dict_ema'] if ema else torch.load(path)['state_dict']
    model.load_state_dict(ckpt)

    return model, config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

parser = argparse.ArgumentParser(description='Deep KSVD-VAE')
# General
parser.add_argument('--data-folder', type=str, default='./dataset',
                        help='name of the data folder')
parser.add_argument('--dataset', type=str, default='imagenet',
                        help='name of the dataset (mnist, fashion-mnist, cifar10, '
                             'miniimagenet, FFHQ, LSUN-cat, LSUN-bedroom, LSUN-church, imagenet)')
#FFHQ
#parser.add_argument('--model-config', type=str, default='./configs/ffhq/stage1/ffhq256-deepksvdvae16x16.yaml')
#IMAGENET
parser.add_argument('--model-config', type=str, default='./configs/imagenet256/stage1/im256-deepksvdvae32x32.yaml')
#parser.add_argument('--model-config', type=str, default='./configs/imagenet256/stage1/im256-deepksvdvae16x16.yaml')
parser.add_argument('--dir_logs', type=str, default='./results/logs')
parser.add_argument('--dir_models', type=str, default='./results/models')


parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers (default: {0})'.format(0))
parser.add_argument('--device', type=str, default='cuda',
                        help='set the device (cpu or cuda, default: cpu)')
parser.add_argument('--steps', type=int, default=0,
                        help='steps recorder for training and test')

#parser.add_argument('-r', '--result_path', type=str, default='./results.tmp')
parser.add_argument('-l', '--load-path', type=str, default='')
parser.add_argument('-p', '--postfix', type=str, default='')
parser.add_argument('--seed', type=int, default=0)

#Distribution Training Parameters
parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--timeout', type=int, default=86400, help='time limit (s) to wait for other nodes in DDP')

parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', action='store_true')

args, extra_args = parser.parse_known_args()

model_config = setup(args, extra_args)
distenv = model_config.runtime.distenv

if __name__ == '__main__':
    import os

    # Create logs and models folder if they don't exist
    '''
    if not os.path.exists(args.dir_logs):
        os.makedirs(args.dir_logs)
    if not os.path.exists(args.dir_models):
        os.makedirs(args.dir_models)
    '''
    # Device
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    #args.device = torch.device('cuda:2')
    main(args, model_config)
