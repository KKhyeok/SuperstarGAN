from model import Generator
from model import Discriminator
from model import Classifier      
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms as T   
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, celeba_class_loader, afhq_loader, afhq_class_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.celeba_class_loader = celeba_class_loader
        self.afhq_loader = afhq_loader
        self.afhq_class_loader = afhq_class_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.c_conv_dim = config.c_conv_dim   
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.c_repeat_num = config.c_repeat_num    
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.c_lr = config.c_lr    
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.c_beta1 = config.c_beta1
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'AFHQ']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
            self.C = Classifier(self.image_size, self.c_conv_dim, self.c_dim, self.c_repeat_num )  
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.c_lr, [self.c_beta1, self.beta2])       
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.C, 'C')     
            
        self.G.to(self.device)
        self.D.to(self.device)
        self.C.to(self.device)      

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(resume_iters)) 
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))    

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr, c_lr):          
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.c_optimizer.param_groups:
            param_group['lr'] = c_lr               
            
    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.c_optimizer.zero_grad() 

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':             
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = 1  
            elif dataset == 'AFHQ':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)    

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)       
        elif dataset == 'AFHQ':
            return F.cross_entropy(logit, target)
        
    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
            data_loader_class = self.celeba_class_loader
        
        elif self.dataset == 'AFHQ':
            data_loader = self.afhq_loader
            data_loader_class = self.afhq_class_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)        
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)  
        data_iter_class = iter(data_loader_class)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        c_lr = self.c_lr       

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)
                
            try:
                x_real_class, label_org_class = next(data_iter_class)
            except:
                data_iter_class = iter(data_loader_class)
                x_real_class, label_org_class = next(data_iter_class)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0)) 
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'AFHQ':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)
                
            x_real = x_real.to(self.device)           # Input images.
            x_real_class = x_real_class.to(self.device)  
            
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.
            label_org_class = label_org_class.to(self.device)
            
            # =================================================================================== #
            #                             2-0. Train the Classifier                               #
            # =================================================================================== #

            # Compute loss with real images.
            out_cls = self.C(x_real_class)
            
            c_loss = self.classification_loss(out_cls, label_org_class, self.dataset)
                      
            self.reset_grad()
            c_loss.backward(retain_graph=True)
            self.c_optimizer.step()

            # Logging.
            loss = {}
            loss['C/loss'] = c_loss.item()                 
            
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src = self.D(x_real)
            #d_loss_real = - torch.mean(out_src)
            d_loss_real = torch.mean(F.relu(1. - torch.mul(out_src, 1.0)))
            
            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)                      
            out_src = self.D(x_fake.detach())
            d_loss_fake = torch.mean(F.relu(1. - torch.mul(out_src, -1.0)))

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_gp'] = d_loss_gp.item()
                        
            # =================================================================================== #
            #                               3. Train the generator                                # 
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:          
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                out_cls_f = self.C(x_fake)
                c_loss_f = self.classification_loss(out_cls_f, c_trg, self.dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * c_loss_f
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = self.lambda_rec * g_loss_rec.item()
                loss['G/loss_cls'] = self.lambda_cls * c_loss_f.item()
                
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)
                        
            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))
                    
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(i+1))   
 
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.C.state_dict(), C_path)     
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                c_lr -= (self.c_lr / float(self.num_iters_decay))   
                self.update_lr(g_lr, d_lr, c_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}, c_lr: {}.'.format(g_lr, d_lr, c_lr))
                
                
    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'AFHQ':
            data_loader = self.afhq_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
                # Translate images.
                x_fake_list = [] 
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))
                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))


class HingeLoss(torch.nn.Module):

    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):

        hinge_loss = 1. - torch.mul(output, target)
        return torch.mean(F.relu(hinge_loss))
                

                
                    
            

                



   