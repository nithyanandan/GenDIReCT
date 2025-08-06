import numpy as np
import copy
import torch
import matplotlib.pyplot as plt
import ehtim as eh
import scipy

class Metrics():
    def __init__(self, gt_img, test_img, clObj, imgdim=64, avg_timescale=600, psize=None):
        # EHT Image Objects
        self.clObj = clObj
        _, self.gt_img = self.match_img_obs(self.clObj.obslist[0], gt_img)
        _, self.test_img = self.match_img_obs(self.clObj.obslist[0], test_img)


        self.imgdim = imgdim
        self.avg_timescale = avg_timescale
        if psize is None:
            self.psize = self.clObj.psize
        else:
            self.psize = psize


    def update_clObj(self, ttype='direct'):
        new_obs = self.gt_img.observe_same_nonoise(self.clObj.obslist[0], ttype=ttype)
        self.clObj.obslist[0] = new_obs
        self.clObj.set_class_quantities_from_obslist(ehtimAvg=self.clObj.ehtimAvg, avg_timescale=self.avg_timescale)
        self.clObj.replace_obs_vis(torch.tensor(self.gt_img.imarr().astype(np.float32)).unsqueeze(0), xfov=self.clObj.psize*206265*1e6*self.test_img.imarr().shape[-2], yfov=self.clObj.psize*206265*1e6*self.test_img.imarr().shape[-1], ttype=ttype)
        return
            
    def calc_nxcorr(self, outputs, labels):
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

    def nxcorr(self):
        gt_img = self.gt_img.regrid_image(self.psize*self.imgdim, self.imgdim)
        return gt_img.compare_images(self.test_img, metric='nxcorr')[0][0]

    def match_img_obs(self, obs, img):
        img.mjd = obs.mjd
        img.ra = obs.ra
        img.dec = obs.dec
        img.rf = obs.rf
        return obs, img

    def chisq_lcamp(self):
        obs = self.clObj.obslist[0]
        obs, image = self.match_img_obs(obs, self.test_img)
        metric = obs.chisq(image, dtype='logcamp', ttype='direct')
        return metric

    def chisq_cp(self):
        obs = self.clObj.obslist[0]
        obs, image = self.match_img_obs(obs, self.test_img)
        return obs.chisq(image, dtype='cphase', ttype='direct')

    def chisq_ci(self, plot=False):
        obs = self.clObj.obslist[0]
        obs, image = self.match_img_obs(obs, self.test_img)
        image._imdict['I'] = image.imarr().flatten()/np.sum(image.imarr())
        # image._imdict['I'] = image.imarr().swapaxes(0, 1).flatten()
        truth_ci, uv = self.clObj.FTCI(torch.tensor(image.imarr()).unsqueeze(0), useObs=True, add_th_noise=False, return_uv=True, normwts=None)
        img_ci = self.clObj.FTCI(torch.tensor(image.imarr()).unsqueeze(0), useObs=False, add_th_noise=False, normwts=None)
        ci_sigma = self.clObj.get_CI_MCerror(torch.tensor(image.imarr()).unsqueeze(0), n=1000, useObs=True, th_noise_factor=1)
        
        snr_hist = torch.log10(torch.abs(truth_ci[0])/ci_sigma)
        # mask = np.where(snr_hist > 2)[0]
        # mask = torch.ones_like(snr_hist).bool()
        # ci_sigma[mask] = truth_ci[0][mask]/10**2
        mask = torch.ones_like(snr_hist).bool()

        if plot:
            # SNR Histogram
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.hist(snr_hist.detach().cpu().numpy(), bins='fd', edgecolor='black', linewidth=1.2)

            # CI Scatter
            uv = uv[0][:2][:][:]
            uv = np.sqrt(np.sum(uv**2, axis=0))
            uv = np.sqrt(np.sum(uv**2, axis=0))/1e9
            fig, ax = plt.subplots()
            ax.plot(uv, truth_ci[0], 'd', ms=6)
            ax.errorbar(uv, truth_ci[0], yerr=ci_sigma, fmt='none', ecolor='black', elinewidth=1, capsize=2)
            ax.plot(uv, img_ci[0], 'o', ms=2, c='grey')
            ax.set_xlabel('$\sqrt{|u_1|^2 + |u_2|^2 + |u_3|^2}$ (G$\lambda$)')
            ax.set_ylabel('Closure Invariant')

        # return torch.sum(torch.abs(truth_ci - img_ci)**2).numpy()#/len(truth_ci[0])
        return (torch.sum(((truth_ci[0][mask] - img_ci[0][mask])/ci_sigma[mask])**2)/self.clObj.N_independent_CIs).numpy()
        

    def eff_res(self, max_fwhm = 100, steps = 20, plot=False, init_val = 0):
        gt_img = self.gt_img.regrid_image(self.psize*self.imgdim, self.imgdim)
        target_nxcorr = gt_img.compare_images(self.test_img, metric='nxcorr')[0][0]
        fwhm_list = np.linspace(0, max_fwhm, steps)/206265/1e6
        blurred_images = [gt_img.blur_circ(i) for i in fwhm_list]
        blurred_images = torch.tensor([i.imarr() for i in blurred_images]).unsqueeze(1)
        blurred_nxcorrs = self.calc_nxcorr(torch.tensor(gt_img.imarr()).unsqueeze(0), blurred_images)[1].numpy()
        coeffs = np.polyfit(fwhm_list, blurred_nxcorrs, 3)
        ffit = np.poly1d(coeffs)

        # Plot to test
        if plot:
            fig, ax = plt.subplots()
            ax.plot(fwhm_list*206265*1e6, blurred_nxcorrs)
            ax.plot(fwhm_list*206265*1e6, ffit(fwhm_list))

        return np.max([scipy.optimize.fsolve(lambda x: ffit(x)-target_nxcorr, init_val/206265/1e6)[0]*206265*1e6, 0])

    def dynamic_range(self, effres=None):
        gt_img = self.gt_img.regrid_image(self.gt_img.fovx(), self.imgdim)

        if effres == None:
            effres = self.eff_res(gt_img, self.test_img)

        image = self.test_img.regrid_image(gt_img.fovx(), self.imgdim)
        image = gt_img.align_images([image])[0][0]
        blurred_gt = gt_img.blur_circ(effres/206265/1e6)
        blurred_gt._imdict['I'] = blurred_gt.imvec.flatten()/np.sum(blurred_gt.imvec)
        image._imdict['I'] = image.imvec.flatten()/np.sum(image.imvec)
        max_gt = np.max(gt_img.imarr())
        D = max_gt/np.abs(image.imarr()-blurred_gt.imarr())
        Dq = np.quantile(D, 0.1)
        
        return Dq