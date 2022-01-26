######################################################
### IMPORTANT: ONLY WORKS FOR TORCHVISION >= 0.9.0 ###
######################################################

import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
from abc import ABC, abstractmethod

DATASET_STATS = {
    "imagenet": [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]],
}

class LabeledTransform(ABC):
    @abstractmethod
    def __call__(self, img, params):
        # Transforms an img with params (same as torchvision.transform classes)
        pass
    
    @abstractmethod
    def get_params(self, img):
        pass
    
    @abstractmethod
    def get_target(self, img, params):
        pass

### Subclasses of torch.nn.module (same as tochvision.transform classes) ###

class Normalize(T.Normalize):
    def __init__(self, dataset, **kwargs):
        assert dataset in DATASET_STATS.keys()
        mean, std = DATASET_STATS[dataset]
        super().__init__(mean, std, **kwargs)
        
class UnNormalize(torch.nn.Module):
    def __init__(self, dataset, **kwargs):
        assert dataset in DATASET_STATS.keys()
        super().__init__()
        self.mean, self.std = DATASET_STATS[dataset]
        self.transform = T.Compose([
            T.Normalize(mean=torch.zeros(3),
                        std=1/torch.Tensor(self.std)),
            T.Normalize(mean=-torch.Tensor(self.mean),
                        std=torch.ones(3)),
       ])
    
    def forward(self, tensor):
        return self.transform(tensor)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

### Subclasses of LabeledTransform ###

class RandomResizedCrop(LabeledTransform):
    def __init__(self, *args, **kwargs):
        self.transform = T.RandomResizedCrop(*args, **kwargs)
    
    def __call__(self, img, params):
        return F.resized_crop(img, *params, self.transform.size, self.transform.interpolation)
    
    def get_params(self, img):
        return self.transform.get_params(img, self.transform.scale, self.transform.ratio)
    
    def get_target(self, img, params):
        top, left, height, width = params
        im_width, im_height = img.size
        v_scale, h_scale = height/im_height, width/im_width
        scale = max(v_scale, h_scale)
        return torch.Tensor([top + height/2, left + width/2])*0.03, 20*torch.Tensor([scale])
    
class ColorJitter(LabeledTransform):
    def __init__(self, *args, **kwargs):
        self.transform = T.ColorJitter(*args, **kwargs)
        
    def __call__(self, img, params):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params
        
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        return img
    
    def get_params(self, img):
        return self.transform.get_params(self.transform.brightness, self.transform.contrast, self.transform.saturation, self.transform.hue)
        
    def get_target(self, img, params):
        return torch.Tensor(list(params)[1:])
    
class Compose(LabeledTransform):
    def __init__(self, transforms):
        self.transforms = transforms
        self.labeled_transforms = []
        for transform in self.transforms:
            if isinstance(transform, LabeledTransform):
                self.labeled_transforms.append(transform)
        
    def __call__(self, img, params):
        transforms = []
        i = 0
        for transform in self.transforms:
            if isinstance(transform, LabeledTransform):
                img = transform(img, params[i])
                i += 1
            else:
                img = transform(img)
        return img
    
    def get_params(self, img):
        return [transform.get_params(img) for transform in self.labeled_transforms]
    
    def get_target(self, img, params):
        targets = []
        for i, transform in enumerate(self.labeled_transforms):
            target = transform.get_target(img, params[i])
            if isinstance(target, list) or isinstance(target, tuple):
                targets = targets + list(target)
            else:
                targets.append(target)
        return targets
