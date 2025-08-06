import torch
import numpy as np
import matplotlib.pyplot as plt
import data.ngEHTMetrics as met
import models.unet2dcondition as unet
import data.CI_torch_v2 as CI
import ehtim as eh
from torch import nn
import imagehash
import models.model_ImageConv as ConvModel
from PIL import Image
from models.encoder import Encoder
from models.decoder import Decoder

class GenDIReCT():
    def __init__(self, model_path, autoencoder_path, uvfits_path, baseid=0, imgdim=64, psize=1.7044214966184275e-11, device='cpu'):
        self.device = device
        # self.encoder = torch.load(encoder_path, weights_only=False, map_location=device).module.to(device)
        self.encoder = Encoder(1, 4, n_res_layers=2, res_h_dim=8).to(device)
        self.decoder = Decoder(4, 4, n_res_layers=2, res_h_dim=8).to(device)

        state_dict = torch.load(autoencoder_path, map_location=device)
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder' in k}
        decoder_state_dict = {k.replace('decoder.', ''): v for k, v in state_dict.items() if 'decoder' in k}
        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)

        obs = eh.obsdata.load_uvfits(uvfits_path)
        self.clObj = CI.Closure_Invariants(obslist = [obs], device=device, ttype='DFT', baseid=baseid, psize=psize)

        self.imgdim = imgdim
        self.psize = psize

        im = np.zeros((1, imgdim, imgdim))
        im[0, imgdim//2, imgdim//2] = 1.0
        data_dim = self.clObj.FTCI(im).shape[-1]
        self.data_dim = data_dim

        self.model = unet.UNet2DCondition(ci_dim=data_dim, model_choice=1, encoder_hid_dim=0).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))

        self.model.eval()
        self.encoder.eval()
        self.decoder.eval()

    def image(self, img, N_images=1024, useObs=False, add_th_noise=False, th_noise_factor=1,
              add_bl_noise=False, bl_noise_factor=0.1,
              interpFactor=1, psizeFactor=1, check_convergence=False,
              plot_clusters=False, verbose=False, plot=True, evalclObj=None,
              evaluate=True):
        if useObs:
            im = torch.zeros((1, 1, self.imgdim, self.imgdim))
        else:
            im = torch.tensor(img.imarr().astype(np.float32)).reshape(1, 1, self.imgdim, self.imgdim)
        x, invs = self.runDiffusion(im, N_images=N_images, useObs=useObs, add_th_noise=add_th_noise, th_noise_factor=th_noise_factor, 
                                    add_bl_noise=add_bl_noise, bl_noise_factor=bl_noise_factor, plot_clusters=plot_clusters)
        invs = invs[0].reshape(1, invs.shape[-1]).cpu()
        ci_sigmas = self.clObj.get_CI_MCerror(im, n=1000, useObs=useObs, th_noise_factor=th_noise_factor)
        self.runCNN(x, invs, ci_sigmas, interpFactor=interpFactor, psizeFactor=psizeFactor, check_convergence=check_convergence, verbose=verbose)
        if evaluate:
            if evalclObj is None:
                self.evaluate(img, useObs=useObs, plot=plot)
            else:
                self.evaluate(img, useObs=useObs, plot=plot, clObj=evalclObj)
            

    def evaluate(self, img, useObs=False, plot=False, print=False, clObj=None):
        res = self.convmodel().detach().cpu().numpy()
        out_res = img.copy()
        out_res = out_res.regrid_image(self.psize*self.imgdim, self.imgdim)
        out_res._imdict['I'] = res.flatten()/res.sum()
        
        if clObj is None:
            metrics = met.Metrics(img, out_res, self.clObj)
        else:
            metrics = met.Metrics(img, out_res, clObj)

        if not useObs:
            metrics.update_clObj(ttype='direct')

        chi2_cphase = metrics.chisq_cp()
        chi2_lcamp = metrics.chisq_lcamp()
        chi2_ci = metrics.chisq_ci(plot=False)
        res_nxcorr = metrics.nxcorr()
        effres = metrics.eff_res(plot=False)
        dynrange = metrics.dynamic_range(effres)

        if print:
            print('Cphase chi2: %.2f' % chi2_cphase)
            print('Logcamp chi2: %.2f' % chi2_lcamp)
            print('CI chi2: %.2f' % chi2_ci)
            print('NXCorr: %.3f' % res_nxcorr)
            print('Effective resolution: %.1f $\\mu$as' % effres)
            print('Dynamic range: %.1f' % dynrange)


        if plot:
            fig, ax =plt.subplots(1, 2, figsize=(9.9, 5))
            fig.subplots_adjust(hspace=0.01, wspace=0.0)
            regrid_img = img.regrid_image(self.psize*self.imgdim, self.imgdim)
            regrid_img.display(axis=ax[0], cfun='afmhot', has_title=False, has_cbar=False)
            ax[1].imshow(res, cmap='afmhot', interpolation='gaussian')
            ax[1].text(0.5, 0.9, '$\\rho_{\\rm{NX}} =$ %.3f\n$\\theta_{\\rm{eff}} =$ %.1f $\\mu$as' % (res_nxcorr, effres), ha='center', va='center', fontsize=8, transform=ax[1].transAxes, c='white')
            ax[1].text(0.5, 0.1, '$\\mathcal{D}_{0.1} =$ %.1f' % (dynrange), ha='center', va='center', fontsize=8, transform=ax[1].transAxes, c='white')
            ax[1].text(0.97, 0.5, '$\\chi^2_{\\rm{cphase}} =$ %.2f\n$\\chi^2_{\\rm{lcamp}} =$ %.2f\n$\\chi^2_{\\rm{ci}} =$ %.2f' % (chi2_cphase, chi2_lcamp, chi2_ci), ha='right', va='center', fontsize=8, transform=ax[1].transAxes, c='white')
            for i, a in enumerate(ax.flatten()):
                a.axis('off')
            plt.show()


    def runDiffusion(self, im, N_images=1024, useObs=False, add_th_noise=False, th_noise_factor=1, 
                     add_bl_noise=False, bl_noise_factor=0.1, plot_clusters=False):
        x = torch.randn(N_images, 4, 16, 16).to(self.device).to(torch.float32)

        invs = self.clObj.FTCI(im, useObs=useObs, add_th_noise=add_th_noise, th_noise_factor=th_noise_factor,
                               add_bl_noise=add_bl_noise, bl_noise_factor=bl_noise_factor)
        invs = invs.to(self.device).repeat(N_images, 1)
        invs = invs.reshape(-1, 1, invs.shape[-1])
        invs = invs.to(torch.float32)

        x = self.model.runUnet(x, invs, init_t=0, guidance_scale=None)
        x = self.decoder(x.to(torch.float32))
        x, _, _ = self.weighted_mean_image(x.cpu(), invs.cpu(), self.clObj)

        x = self.shift_all(x[0].detach().cpu(), x.detach().cpu())

        if plot_clusters:
            clusters = self.findClusters(x, threshold=4, hashsize=6, type=0, verbose=True)

            ngrid = int(np.sqrt(len(clusters)))+1
            fig, axs = plt.subplots(ngrid, ngrid, figsize=(12, 12))
            fig.subplots_adjust(hspace=0, wspace=0)
            axs = axs.flatten()
            for a in axs:
                a.axis('off')

            for ind, i in enumerate(clusters):
                PIL_imgs = clusters[i]
                c_images = [torch.tensor(np.asarray(i)) for i in PIL_imgs]
                c_rep = torch.tensor(c_images[0])
                c_images = self.shift_all(c_rep, c_images)
                c_invs = invs[:len(c_images)]
                c_images, c_mean, ciloss = self.weighted_mean_image(torch.tensor(c_images).unsqueeze(1), c_invs.cpu(), self.clObj)

                axs[ind].imshow(c_mean, cmap='Greys', interpolation='gaussian')
                axs[ind].text(0.5, 0.95, 'Cluster ' + str(i) + ': \n' + str(len(c_images)), ha='center', va='center', fontsize=8, transform=axs[ind].transAxes,)
            plt.show()
            
        return x, invs

    def runCNN(self, x, ci, ci_sigmas, interpFactor=1, psizeFactor=1, check_convergence=False, verbose=False):
        self.convmodel = ConvModel.ImageConv(x, ci, ci_sigmas, self.clObj, interpF=lambda x: x/x.max(), device=self.device, interpFactor=interpFactor, psizeFactor=psizeFactor).to(self.device)
        self.convmodel.train(nepochs=1000, init_lr=1e-3, decay=0.999, verbose=verbose, loss_type='L1', weighting=True, loss_reduction='mean', optimiser='Adam', check_convergence=check_convergence, suppress_out=not verbose)
        self.convmodel.train(nepochs=2000, init_lr=1e-4, decay=0.999, verbose=verbose, loss_type='L2', loss_reduction='sum', optimiser='Adam', check_convergence=check_convergence, suppress_out=not verbose)
        # self.convmodel.eval()
        return self.convmodel.train_images


    def weighted_mean_image(self, imgs, invs, clObj):
        recon_ci = clObj.FTCI_batch(64, imgs.reshape(-1, imgs.shape[-2], imgs.shape[-1])).reshape(-1, 1, invs.shape[-1])
        # recon_ci = clObj.FTCI(imgs.reshape(-1, imgs.shape[-2], imgs.shape[-1])).reshape(-1, 1, invs.shape[-1])
        loss = nn.L1Loss(reduction='none')

        ciloss = loss(invs, torch.Tensor(recon_ci))
        civar = ciloss.reshape(-1, invs.shape[-1]).var(dim=1)
        sse = (ciloss.reshape(-1, invs.shape[-1])**2).sum(dim=1)

        imgs = imgs.reshape(-1, 1, imgs.shape[-2], imgs.shape[-1])
        imgs = imgs[torch.argsort(sse)] # images sorted by loss

        weighted_image = torch.zeros_like(imgs[0])
        for i in range(imgs.shape[0]):
            weighted_image += imgs[i] * 1/civar[i]
        weighted_image /= civar.sum()

        return imgs, weighted_image.detach().cpu().numpy()[0], sse

    def nxcorr(self, outputs, labels):
        dim = int(outputs.shape[-1])
        outputs = outputs.reshape(-1, dim**2)
        labels = labels.reshape(-1, dim**2)
        
        outputs_norm = (outputs.reshape(-1, dim, dim) - torch.nanmean(outputs, axis=1).reshape(-1, 1, 1)) / torch.std(outputs, axis=1).reshape(-1, 1, 1)
        labels_norm = (labels.reshape(-1, dim, dim) - torch.nanmean(labels, axis=1).reshape(-1, 1, 1)) / torch.std(labels, axis=1).reshape(-1, 1, 1)

        fft_outputs = torch.fft.fftn(outputs_norm, s=[outputs_norm.size(d)*1 for d in [1,2]], dim=[1,2])
        fft_labels = torch.fft.fftn(labels_norm, s=[outputs_norm.size(d)*1 for d in [1,2]], dim=[1,2])

        xcorr = torch.fft.ifftn(fft_outputs * torch.conj(fft_labels), dim=[1,2])

        nxcorr_flat = xcorr.reshape(-1, dim**2)
        idx = torch.argmax(torch.abs(nxcorr_flat), dim=1)

        return idx, torch.abs(nxcorr_flat[torch.arange(nxcorr_flat.shape[0]), idx])/dim**2

    def shift_image(self, im1, im2): # shift single im2 by idx
        idx, _ = self.nxcorr(im1, im2)
        im2 = torch.roll(im2, shifts=int(idx))
        return im1, im2

    def shift_all(self, truth, imgs):
        shifted_imgs = []
        for img in imgs:
            _, shifted_img = self.shift_image(truth, img)
            shifted_imgs.append(shifted_img)
        return np.array(shifted_imgs)

    def ecdf(self, a):
        x, counts = np.unique(a, return_counts=True)
        cusum = np.cumsum(counts)
        return x, cusum / cusum[-1]

    def crps_score(self, truth, imgs):
        imgs = np.array(imgs)
        imgs = imgs.reshape(imgs.shape[0], -1)
        truth = truth.reshape(-1)

        imgs = np.array([i/np.sum(i) for i in imgs])
        truth = truth/np.sum(truth)
        crps_scores = []
        for i in range(imgs.shape[1]):
            out_pix = imgs[:, i]
            # mean_out_pix = np.mean(out_pix)+0.1
            truth_pix = truth[i]
            x, y = self.ecdf(out_pix)
            step_fc = np.heaviside(x-truth_pix, 0.5)
            crps = np.trapz((y-step_fc)**2, x)
            crps_scores.append(crps)

        crps_scores = np.array(crps_scores).reshape(64, 64)
        tot_crps = np.sum(crps_scores)/(truth.shape[0])
        return tot_crps, crps_scores    

    def plot_images(self, images, cmap='viridis', return_axes=False, show=True):
        fig, axes = plt.subplots(2, len(images)//2, figsize=(len(images)//2*0.99, 2))
        fig.subplots_adjust(hspace=0., wspace=0.)
        axes = axes.flatten()
        for ax, img in zip(axes, images):
            ax.imshow(img.permute(1, 2, 0), cmap=cmap)
            ax.axis('off')
        if show:
            plt.show()
        if return_axes:
            return axes

    def findClusters(self, x, threshold=4, hashsize=6, type=0, verbose=False):
        # hash all images
        clusters = {}
        for img in x:
            PIL_img = Image.fromarray(img[0]/np.max(img[0])*255)
            hash = imagehash.phash(PIL_img, hash_size=hashsize)
            clusters[hash] = clusters.get(hash, []) + [PIL_img]

        # group all hashes that are only different by threshold
        if type == 0: # Clusters by distance from hashes ordered by L1Loss of closure invariants
            moved_hashes = []
            for ind, hash in enumerate(clusters.keys()):
                if ind != len(clusters.keys()) - 1 and hash not in moved_hashes:
                    test = np.array([hash - b for b in list(clusters.keys())[ind+1:]])
                    for i in np.array(list(clusters.keys())[ind+1:])[np.where(test <= threshold)]:
                        if i not in moved_hashes:
                            moved_hashes.append(i)
                            clusters[hash] += clusters[i]
            
            for i in moved_hashes:
                if i in clusters.keys():
                    del clusters[i]
        else:
            grouping = {} # Clusters by continuous hash connections
            for ind, hash in enumerate(clusters.keys()):
                if ind != len(clusters.keys()) - 1:
                    test = np.array([hash - b for b in list(clusters.keys())[ind+1:]])
                    for i in np.array(list(clusters.keys())[ind+1:])[np.where(test <= threshold)]:
                        if i not in grouping:
                            grouping[i] = hash

            for i in dict(reversed(list(clusters.items()))):
                if i in grouping:
                    clusters[grouping[i]] += clusters[i]
                    clusters[i] = []

            empty_keys = []
            for i in clusters:
                if len(clusters[i]) == 0:
                    empty_keys.append(i)
            for i in empty_keys:
                del clusters[i]

        if verbose:
            print('Number of Clusters: ' + str(len(clusters)))

        return clusters

    def ordered_hash(self, x, target, hashsize=12, cmap='Greys', fide_type='nxcorr', plot=False, num_images = 10):
        PIL_img = Image.fromarray(target/np.max(target)*255)
        target_hash = imagehash.phash(PIL_img, hash_size=hashsize)
        ordered_images = []
        for img in x:
            PIL_img = Image.fromarray(img[0]/np.max(img[0])*255)
            if fide_type == 'hash':
                hash = imagehash.phash(PIL_img, hash_size=hashsize)
                ordered_images.append([1 - int(hash-target_hash)/hashsize**2, PIL_img])
            elif fide_type == 'nxcorr':
                xcorr = self.nxcorr(torch.tensor(img), torch.tensor(target))[1]
                ordered_images.append([xcorr, PIL_img])
            else:
                ordered_images.append([0, None])
        ordered_images = sorted(ordered_images, key=lambda x: x[0])[::-1]
        if plot:
            # flattened images
            images = ordered_images[:num_images]
            fig, axes = plt.subplots(2, len(images)//2, figsize=(len(images)//2*0.99, 2))
            fig.subplots_adjust(hspace=0., wspace=0.)
            axes = axes.flatten()
            for ax, data in zip(axes, images):
                ax.imshow(data[1], cmap=cmap)
                ax.axis('off')
                ax.text(0.5, 0.9, '%.3f' % data[0], ha='center', va='center', fontsize=8, transform=ax.transAxes, c='white')
        return ordered_images
