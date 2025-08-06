from torchvision.transforms import v2
import torchvision
import torch
import numpy as np

class imgTransforms():
    def __init__(self):
        # random resizing during training
        # albumentation
        # kornia
        # affine transformation (zoom in zoom out, rotate, shear) elastic deformation/warping

        self.train_transforms = v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            self.flip_xy,
            v2.RandomAffine(degrees=(-90, 90), scale=(0.8, 1.), shear=(-5, 5), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            v2.CenterCrop(64),
        ])

        self.val_transforms = v2.Compose([
            v2.ToTensor(),
            v2.CenterCrop(64),
        ])

        self.val_transforms_rotate = v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            self.flip_xy,
            v2.RandomAffine(degrees=(-90, 90), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            v2.CenterCrop(64),
        ])


        self.val_transforms_taper = v2.Compose([
            v2.ToTensor(),
            self.fixed_taper,
            v2.CenterCrop(64),
        ])


        self.cifar_train_transforms = v2.Compose([
            v2.ToTensor(),
            v2.Resize(64),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            self.flip_xy,
            v2.RandomAffine(degrees=(-90, 90), scale=(0.8, 1.), shear=(-5, 5), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            self.radial_taper,
            v2.CenterCrop(64),
        ])

        self.cifar_val_transforms = v2.Compose([
            v2.ToTensor(),
            v2.Resize(64),
            self.fixed_taper,
            v2.CenterCrop(64),
        ])

        self.cifar_train_transforms_v2 = v2.Compose([
            v2.ToTensor(),
            # v2.Pad(128),
            # v2.Resize(96),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            self.flip_xy,
            v2.RandomAffine(degrees=(-90, 90), shear=(-30, 30), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            self.radial_taper_v2,
            # v2.GaussianBlur(15, sigma=(0.01, 3)),
            # v2.CenterCrop(64),
        ])

        self.train_transforms_v2 = v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            self.flip_xy,
            v2.RandomAffine(degrees=(-90, 90), scale=(0.8, 1.), shear=(-5, 5), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            # v2.CenterCrop(64),
        ])

        self.val_transforms_v2 = v2.Compose([
            v2.ToTensor(),
            # v2.CenterCrop(64),
        ])


        self.test_transforms = v2.Compose([
            v2.ToTensor(),
            # v2.Resize(38),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            self.flip_xy,
            v2.RandomAffine(degrees=(-90, 90), scale=(0.4, 0.7), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            # v2.RandomAffine(degrees=(0, 0), scale=(0.62, 0.62), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            # radial_taper,
            v2.CenterCrop(64),
        ])

        self.test_transforms_noRotate = v2.Compose([
            v2.ToTensor(),
            # v2.Resize(38),
            v2.RandomAffine(degrees=(0, 0), scale=(0.4, 0.7), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            # v2.RandomAffine(degrees=(0, 0), scale=(0.62, 0.62), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            # radial_taper,
            v2.CenterCrop(64),
        ])


        self.test_transforms_noise = v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            self.flip_xy,
            v2.RandomAffine(degrees=(-90, 90), scale=(0.4, 0.7), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            v2.CenterCrop(64),
            self.gauss_noise_tensor,
        ])

        self.fidelity_test = v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            self.flip_xy,
            v2.RandomAffine(degrees=(-90, 90), scale=(0.7, 1.3), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            v2.CenterCrop(64),
        ])

        self.fidelity_test_shear = v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            self.flip_xy,
            v2.RandomAffine(degrees=(-90, 90), scale=(0.7, 1.3), shear=(-30, 30), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            v2.CenterCrop(64),
        ])      

        self.combination_transforms = v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            self.flip_xy,
            v2.RandomAffine(degrees=(-90, 90), scale=(0.8, 1.0), translate=(0.1, 0.1), shear=(-30, 30), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        ]) 

        self.tiny_transforms = v2.Compose([
            v2.ToTensor(),
            v2.RandomAffine(degrees=(-5, 5), scale=(0.95, 1.05), shear=(-20, 20), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            # v2.CenterCrop(64),
        ])

        self.tiny_rotate = v2.Compose([
            v2.ToTensor(),
            v2.RandomAffine(degrees=(-90, 90), scale=(0.95, 1.05), shear=(-5, 5), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            # v2.CenterCrop(64),
        ])


    def gauss_noise_tensor(self, img, sigma=0.1):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)
        
        out = img + sigma * torch.randn_like(img)
        
        if out.dtype != dtype:
            out = out.to(dtype)
            
        return out

    def square_taper(self, img, square_size_range=(32, 48), blur_fwhm_range=(5, 10), fixed=False):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)
        
        h, w = img.shape[-2:]
        if fixed:
            square_size = square_size_range
            blur_fwhm = blur_fwhm_range
        else:
            square_size = np.random.randint(*square_size_range)
            blur_fwhm = np.random.uniform(*blur_fwhm_range)

        # taper mask
        taper = torch.zeros_like(img)
        taper[:, h//2-square_size//2:h//2+square_size//2, w//2-square_size//2:w//2+square_size//2] = 1
        taper = v2.GaussianBlur(25, sigma=blur_fwhm)(taper)
        
        out = img * taper
        
        if out.dtype != dtype:
            out = out.to(dtype)
            
        return out

    def radial_taper(self, img, radii_range=(8, 24), blur_fwhm_range=(5, 10), fixed=False):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)
        
        h, w = img.shape[-2:]
        if fixed:
            radius = radii_range
            blur_fwhm = blur_fwhm_range
        else:
            radius = np.random.randint(*radii_range)
            blur_fwhm = np.random.uniform(*blur_fwhm_range)

        # radial taper mask
        taper = torch.zeros_like(img)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        y = y - h//2
        x = x - w//2
        r = torch.sqrt(x**2 + y**2)
        taper[:, r<radius] = 1
        taper = v2.GaussianBlur(15, sigma=blur_fwhm)(taper)

        
        out = img * taper
        
        if out.dtype != dtype:
            out = out.to(dtype)
            
        return out

    def radial_taper_v2(self, img, radii_range=(3, 24), blur_fwhm_range=(10, 20), fixed=False):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)
        
        h, w = img.shape[-2:]
        if fixed:
            radius = radii_range
            blur_fwhm = blur_fwhm_range
        else:
            radius = np.random.randint(*radii_range)
            blur_fwhm = np.random.uniform(*blur_fwhm_range)

        # radial taper mask
        taper = torch.zeros_like(img)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        y = y - h//2
        x = x - w//2
        r = torch.sqrt(x**2 + y**2)
        # taper rapidly drop to 0 for r > radius
        taper[:, r>radius] = (1 - (r[r>radius] - radius) / (r.max()//2 - radius)*3).to(dtype)
        taper[:, r<radius] = 1
        # non negative
        taper[taper < 0] = 0


        taper = v2.GaussianBlur(15, sigma=blur_fwhm)(taper)
        out = img * taper
        
        if out.dtype != dtype:
            out = out.to(dtype)
            
        return out

    def fixed_taper(self, img, size = 16, blur_fwhm = 5):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)

        out = self.radial_taper(img, size, blur_fwhm, fixed=True)
        

        if out.dtype != dtype:
            out = out.to(dtype)
            
        return out

    def flip_xy(self, img, p=0.5):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)

        if torch.rand(1) < p:
            img = torch.swapaxes(img, -1, -2)

        if img.dtype != dtype:
            img = img.to(dtype)

        return img
    
    def centerCrop(self, imgs, size=64):
        if isinstance(imgs, torch.Tensor):
            return v2.CenterCrop(size)(imgs)
        elif isinstance(imgs, list):
            return [v2.CenterCrop(size)(img) for img in imgs]
        else:
            raise ValueError("Input should be either torch.Tensor or list of torch.Tensor")
