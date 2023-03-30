import os
import argparse
from solver import Solver
from data_loader import get_loader, get_loader_class
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
        
    # Data loader.
    celeba_loader = None
    celeba_class_loader = None
    afhq_loader = None
    afhq_class_loader = None
    
    if config.dataset == 'CelebA':
        celeba_loader = get_loader(config.celeba_image_dir, None, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
        celeba_class_loader = get_loader_class(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)       
    if config.dataset == 'AFHQ':
        afhq_loader = get_loader(config.afhq_image_dir, config.afhq_test_dir,  None, None,  512,  config.image_size, config.batch_size,
                                 'AFHQ', config.mode, config.num_workers)
        afhq_class_loader = get_loader_class(config.afhq_image_dir, None, None,  512, config.image_size, config.batch_size,
                                 'AFHQ', config.mode, config.num_workers)    

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, celeba_class_loader,afhq_loader, afhq_class_loader, config)

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'AFHQ']:
            solver.train()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'AFHQ']:
            solver.test()
       
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=40, help='dimension of domain labels dataset')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--c_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D') 
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--c_repeat_num', type=int, default=6, help='number of strided conv layers in C')     
    parser.add_argument('--lambda_cls', type=float, default=0.25, help='weight for domain classification loss')  
    parser.add_argument('--lambda_rec', type=float, default=1.3, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=1, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'AFHQ'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--c_lr', type=float, default=0.00012, help='learning rate for C')      
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--c_beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['5_o_Clock_Shadow', 'Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair',
                        'Blurry','Brown_Hair','Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard','Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=1000000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--afhq_image_dir', type=str, default='data/afhq/train')
    parser.add_argument('--afhq_test_dir', type=str, default='data/afhq/test')
    parser.add_argument('--log_dir', type=str, default='superstarGAN/afhq/logs')
    parser.add_argument('--model_save_dir', type=str, default='superstarGAN/afhq/models')
    parser.add_argument('--sample_dir', type=str, default='superstarGAN/afhq/samples')
    parser.add_argument('--result_dir', type=str, default='superstarGAN/afhq/results')
    
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()  
    print(config)
    main(config)


    