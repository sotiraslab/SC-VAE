import numpy as np
import torch
torch.set_printoptions(precision=10)
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from scipy import linalg
from six.moves import urllib
from img_datasets import create_dataset
import argparse
import math
from scvae.utils.setup import setup
from scvae.models import scvae
from scvae.optimizer import create_optimizer, create_scheduler

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import time
from tensorboardX import SummaryWriter


def train(data_loader, model, optimizer, scheduler, args, writer):
    for images, images_with_noise in data_loader:
        images = images.to(args.device)

        optimizer.zero_grad()
        x_tilde, rec_latent_representation, Dictionary, z, _, mincutpoolloss, ortholoss = model(images)
        sparsity_nonzero, sparsity_nmf = look_sparsity(z)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Latent Sparse dictionary learning objective
        loss_sdl = torch.mean(rec_latent_representation[0]*rec_latent_representation[1])

        if mincutpoolloss is not None:
            loss = loss_recons + loss_sdl + (mincutpoolloss + ortholoss)
            print('loss:' + str(loss.cpu()) + ' loss_recons:' + str(loss_recons.cpu()) +
                  ' loss_sdl:' + str(loss_sdl.cpu()) + 'mincutpoolloss:' + str(mincutpoolloss) + 'ortholoss:' + str(
                ortholoss) + ' sparsity_nonzero:' + str(sparsity_nonzero), flush=True)
            writer.add_scalar('loss/train/loss_mincutpool', mincutpoolloss, args.steps)
            writer.add_scalar('loss/train/ortholoss', ortholoss, args.steps)
            writer.add_scalar('loss/train/ortholoss and mincutpoolloss', ortholoss + mincutpoolloss, args.steps)
        else:
            loss = loss_recons + loss_sdl
            print('loss:' + str(loss.cpu()) + ' loss_recons:' + str(loss_recons.cpu()) +
                  ' loss_sdl:' + str(loss_sdl.cpu()) + ' sparsity_nonzero:' + str(sparsity_nonzero), flush=True)
        loss.backward()

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/train/sparse dictionary learning', loss_sdl.item(), args.steps)
        writer.add_scalar('loss/test/sparsity_nonzero', sparsity_nonzero, args.steps)
        writer.add_scalar('loss/test/sparsity_nmf', sparsity_nmf, args.steps)

        optimizer.step()
        scheduler.step()

        args.steps += 1

def test(data_loader, model, args, writer):

    with torch.no_grad():
        loss_recons, loss_sdl = 0., 0.
        for images, _ in data_loader:
            images = images.to(args.device)
            x_tilde, rec_latent_representation, Dictionary, z, _, _, _ = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_sdl += torch.mean(rec_latent_representation[0] * rec_latent_representation[1])
            sparsity_nonzero, sparsity_nmf = look_sparsity(z)
            print('sparsity_nonzero:' + str(sparsity_nonzero))
            print('sparsity_nmf:' + str(sparsity_nmf))

        loss_recons /= len(data_loader)
        loss_sdl /= len(data_loader)

    # Logs
    writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
    writer.add_scalar('loss/test/sparse dictionary learning', loss_sdl.item(), args.steps)
    writer.add_scalar('loss/test/sparsity_nonzero', sparsity_nonzero, args.steps)
    writer.add_scalar('loss/test/sparsity_nmf', sparsity_nmf, args.steps)

    return loss_recons.item(), loss_sdl.item(), Dictionary

def look_sparsity(z):

    total_number = z.numel()
    total_nonzero = z.detach().count_nonzero()
    sparsity_nonzero = (total_number-total_nonzero)/total_number

    sparsity_nmf = calculate_sparsity(z.detach().cpu().numpy().reshape(-1))

    return sparsity_nonzero, sparsity_nmf

def calculate_sparsity(W):
    D = np.shape(W)[0]
    numerator = np.sum( np.abs(W), axis=0)
    denominator = np.sqrt(np.sum(np.power(W,2),axis=0))
    subtract_by = np.divide(numerator, denominator)
    del numerator, denominator
    subtract_from = np.sqrt(D)
    subtracted = subtract_from - subtract_by
    del subtract_from, subtract_by
    numerator = np.mean(subtracted)
    del subtracted
    denominator = np.sqrt(D)-1
    sparsity = np.divide( numerator, denominator)
    return sparsity

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, rec_latent_representation, _, _, multi_heads_alphas, _, _ = model(images)
    return x_tilde, rec_latent_representation, multi_heads_alphas

def rec_multi_head_attentions(writer, multi_heads_alphas, num_attention_heads, epoch):
    for i in range(num_attention_heads):
        multi_heads_alpha = multi_heads_alphas[:, i, :, :]
        multi_heads_alpha = multi_heads_alpha.view(multi_heads_alpha.size(0),
                                                     int(multi_heads_alpha.size(1) ** 0.5),
                                                     int(multi_heads_alpha.size(1) ** 0.5),
                                                     multi_heads_alpha.size(2))
        multi_heads_alpha = make_grid(multi_heads_alpha.permute(0, 3, 1, 2).cpu(), nrow=8, range=(-1, 1),
                                     normalize=True)
        writer.add_image('multi_heads_alpha_' + str(i), multi_heads_alpha, epoch)

def main(args, model_config):
    #load dataset
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
    elif args.dataset == 'FFHQ':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'imagenet':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    else:
        raise NotImplementedError('%s not implemented..' % args.dataset)

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=model_config.experiment.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=model_config.experiment.batch_size, shuffle=False, drop_last=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=16, shuffle=False)

    # Fixed images for Tensorboard
    fixed_images, img_with_noise = next(iter(test_loader))

    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)

    date = time.strftime('%Y_%m_%d_%H_%M_%S')
    save_filename = args.dir_models + '/' + args.dataset + '/l_alpha_f_beta_' + args.dataset + '_attention' + args.attention + '_num_epochs' + str(
        model_config.experiment.epochs) + '_bs' + str(model_config.experiment.batch_size) + '_num_soft_thresh' + str(
        model_config.arch.latent.num_soft_thresh) + '_beta' + str(model_config.arch.latent.beta) +\
                 '_' + str(date) + '_r' + str(args.model_config.split('/')[-1].split('vae')[1].split('.')[0])  + '_beta_mincut' + str(model_config.arch.latent.beta_mincut)\
                 + '_num_att_heads' + str(model_config.arch.latent.num_attention_heads)
    print(save_filename); os.makedirs(save_filename)
    logdir = args.dir_logs + '/' + args.dataset + '/l_alpha_f_beta_' + args.dataset + '_attention' + args.attention + '_num_epochs' + str(
            model_config.experiment.epochs) + '_bs' + str(model_config.experiment.batch_size) + '_num_soft_thresh' + str(
            model_config.arch.latent.num_soft_thresh) + '_beta' + str(model_config.arch.latent.beta) +\
                            '_' + str(date) + '_r' + str(args.model_config.split('/')[-1].split('vae')[1].split('.')[0]) + '_beta_mincut' + str(model_config.arch.latent.beta_mincut)\
                + '_num_att_heads' + str(model_config.arch.latent.num_attention_heads)
    print(logdir)
    writer = SummaryWriter(logdir)
    writer.add_image('original', fixed_grid, 0)

    Hidden_size, H_1, H_2, num_soft_thresh = model_config.arch.vae.hidden_size, model_config.arch.alpha.H_1, model_config.arch.alpha.H_2, model_config.arch.latent.num_soft_thresh

    # Initialization of num of atoms
    Dict_init = scvae.init_dct(int(np.sqrt(Hidden_size)), 23)
    num_atoms = min(23*23, model_config.arch.latent.num_atoms)
    Dict_init = Dict_init[:, :num_atoms].to(args.device)


    c_init = (linalg.norm(Dict_init.cpu(), ord=2)) ** 2
    c_init = torch.FloatTensor((c_init,))
    c_init = c_init.to(args.device)

    w_init = torch.normal(mean=1, std=1 / 10 * torch.ones(Hidden_size)).float()
    w_init = w_init.to(args.device)

    model = scvae.Model_VAEf16(
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
        model_config.arch.latent.beta_mincut,
        args.device,
        args.attention,
        model_config.arch.latent.num_attention_heads,
    )
    model.to(args.device)

    num_epochs = model_config.experiment.epochs
    steps_per_epoch = math.ceil(len(train_loader) / (model_config.experiment.batch_size * distenv.world_size))

    if not args.eval:
        optimizer = create_optimizer(model, model_config)
        scheduler = create_scheduler(
            optimizer, model_config.optimizer.warmup, steps_per_epoch,
            model_config.experiment.epochs, distenv
        )

    # Generate the samples first once
    reconstruction, rec_latent_representation, multi_heads_alphas = generate_samples(fixed_images, model, args)
    rec = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('reconstruction', rec, 0)
    alpha = rec_latent_representation[0]
    alpha = make_grid(alpha.permute(0, 3, 1, 2).cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('alpha', alpha, 0)

    if multi_heads_alphas is not None:
        rec_multi_head_attentions(writer, multi_heads_alphas, args.num_attention_heads, 0)

    best_loss = -1.
    for epoch in range(num_epochs):
        print('epoch:' + str(epoch))
        train(train_loader, model, optimizer, scheduler, args, writer)
        loss, _, Dictionary = test(valid_loader, model, args, writer)

        writer.add_image('Dictionary', Dictionary, epoch + 1, dataformats='HW')

        reconstruction, rec_latent_representation, multi_heads_alphas = generate_samples(fixed_images, model, args)
        rec = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        writer.add_image('reconstruction', rec, epoch + 1)

        alpha = rec_latent_representation[0]
        alpha = make_grid(alpha.permute(0, 3, 1, 2).cpu(), nrow=8, range=(-1, 1), normalize=True)
        writer.add_image('alpha', alpha, epoch + 1)

        if multi_heads_alphas is not None:
            rec_multi_head_attentions(writer, multi_heads_alphas, args.num_attention_heads, epoch + 1)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        with open('{0}/last.pt'.format(save_filename), 'wb') as f:
            torch.save(model.state_dict(), f)


parser = argparse.ArgumentParser(description='SC-VAE')
# General
parser.add_argument('--data-folder', type=str, default='./dataset',
                        help='name of the data folder')
parser.add_argument('--dataset', type=str, default='FFHQ',
                        help='name of the dataset (mnist, fashion-mnist, cifar10, '
                             'miniimagenet, FFHQ, LSUN-cat, LSUN-bedroom, LSUN-church)')
parser.add_argument('--model-config', type=str, default='./configs/ffhq/stage1/ffhq256-scvae16x16.yaml')
parser.add_argument('--dir_logs', type=str, default='./results/logs')
parser.add_argument('--dir_models', type=str, default='./results/models')
parser.add_argument('--attention', type=str, default='constant', help='GAT, constant, eq, SSGC, mincutpool')
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
    if not os.path.exists(args.dir_logs):
        os.makedirs(args.dir_logs)
    if not os.path.exists(args.dir_models):
        os.makedirs(args.dir_models)
    # Device
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    #args.device = torch.device('cuda:2')
    main(args, model_config)
