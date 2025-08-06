import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image, ImageDraw


class ImageConv(nn.Module):
        def __init__(self, images, target_ci, ci_sigmas, clObj, interpF = lambda x: x/x.max()*1000, device=torch.device("cpu"), interpFactor=1, psizeFactor=1):
                super().__init__()
                interpFactor *= 4 # account for the convolutional downsampling
                images = F.interpolate(torch.tensor([interpF(i) for i in images]).to(device), scale_factor=(interpFactor,interpFactor), mode='bicubic').squeeze()
                in_dim = len(images)

                self.images = images
                self.target_ci = target_ci
                self.ci_sigmas = ci_sigmas
                self.clObj = clObj
                self.trained = False
                self.train_loss = []
                self.train_images = []
                self.fov = self.clObj.psize*206265*1e6*images.shape[-1]/interpFactor*psizeFactor

                kernel = 4
                stride = 2
                self.conv_stack = nn.Sequential(
                        nn.Conv2d(in_dim, in_dim//2, kernel_size=kernel,
                                stride=stride, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_dim//2, in_dim//4, kernel_size=kernel,
                                stride=stride, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_dim//4, 1, kernel_size=kernel-1,
                                stride=stride-1, padding=1),
                        nn.ReLU(),
                ).to(device)

                # initialise weights
                for m in self.conv_stack.modules():
                        if isinstance(m, nn.Conv2d):
                                nn.init.xavier_uniform_(m.weight)
                                if m.bias is not None:
                                        nn.init.zeros_(m.bias)
                        elif isinstance(m, nn.Linear):
                                nn.init.xavier_uniform_(m.weight)
                                if m.bias is not None:
                                        nn.init.zeros_(m.bias)


        def forward(self):
                x = self.conv_stack(self.images)
                return x.squeeze() 
        
        def train(self, nepochs=2000, init_lr=1e-3, decay=0.999, plot_step=300, weighting=True, loss_type='L2', loss_reduction='mean', optimiser='Adam',
                  check_convergence=False,
                  verbose = False, suppress_out=False):
                
                false_start = False
                l1 = lambda epoch: decay ** epoch

                # Loss selection
                if loss_type == 'L1':
                        criterion = nn.L1Loss(reduction=loss_reduction)
                elif loss_type == 'L2':
                        criterion = nn.MSELoss(reduction=loss_reduction)
                else:
                        criterion = nn.MSELoss(reduction=loss_reduction)
                
                # Optimiser selection
                if optimiser == 'AdamW':
                        optimizer = torch.optim.AdamW(self.conv_stack.parameters(), lr=init_lr)
                else:
                        optimizer = torch.optim.Adam(self.conv_stack.parameters(), lr=init_lr)

                # Scheduler selection
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l1)
        
                for epoch in tqdm(range(nepochs)):
                        self.conv_stack.train()
                        optimizer.zero_grad()
                        outputs = self.forward()
                        out_ci = self.clObj.FTCI(outputs.unsqueeze(0), fov=self.fov)
                        if weighting:
                                loss = criterion(out_ci/self.ci_sigmas, self.target_ci/self.ci_sigmas)
                        else:
                                loss = criterion(out_ci, self.target_ci)

                        if not torch.isnan(loss):
                                loss.backward()
                                optimizer.step()
                                if loss_reduction == 'sum':
                                        loss = loss/self.clObj.N_independent_CIs
                                self.train_loss.append(loss.item())
                                self.train_images.append(self.forward().detach().cpu().numpy())
                                
                                if check_convergence:
                                        if epoch > 100 and np.mean(self.train_loss[-100:]) < 1:
                                                print('Early termination: Mitigate overfitting.')
                                                break
                                        # check if loss is not improving much
                                        if epoch > 200 and np.abs(np.mean(self.train_loss[-100:]) - np.mean(self.train_loss[-200:-100])) < 1e-3:
                                                print('Early termination: Loss convergence.')
                                                break
                        else:
                                print('Retry: Did not initialise well.')
                                false_start = True
                                break
                        
                        if epoch == 0 and not suppress_out:
                                self.conv_stack.eval()
                                res = self.forward().detach().cpu().numpy()
                                fig, ax = plt.subplots()
                                ax.imshow(res, cmap='afmhot')
                                ax.set_title('Init')
                                plt.show()

                        if verbose and epoch % plot_step == 0 and epoch != 0:
                                self.conv_stack.eval()
                                res = self.forward().detach().cpu().numpy()
                                fig, ax = plt.subplots()
                                ax.imshow(res, cmap='afmhot')
                                plt.show()

                        scheduler.step()
                
                self.trained = True
                if verbose and (not false_start) and (not suppress_out):
                        res = self.forward().detach().cpu().numpy()
                        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                        ax[0].imshow(res, cmap='afmhot', interpolation='gaussian')
                        ax[1].plot(self.train_loss)
                        plt.show()
                        
                return


        def make_gif(self, filename='results/refinement.gif', step=10):
                if self.trained and len(self.train_images) > 0:
                        images = []
                        for image in self.train_images:
                                image = Image.fromarray(image/np.max(image)*255)
                                images.append(image)      
                        images[0].save(filename,
                                save_all=True, append_images = images[1::step],
                                optimise=True, duration=2)


                        
