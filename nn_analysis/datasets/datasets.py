import torchvision
import torch
import torchvision.transforms as T
import numpy as np
import os
from abc import ABC, abstractmethod

from nn_analysis import utils
from nn_analysis.constants import ENV_CONFIG_PATH

env_config = utils.load_config(ENV_CONFIG_PATH)

def get_dataset(dataset_name, split='train', **kwargs):
    if dataset_name == 'imagenet':
        dataset = torchvision.datasets.ImageNet(env_config['imagenet_path'], split=split, **kwargs)
        
    elif dataset_name == 'pseudo_hvm':
        # size (64, 6, 5, 256, 256)
        images = torch.from_numpy(np.load(os.path.join(env_config['pseudohvm_path'],'pseudoHVM_factorized_format.npy')))
        images = images.reshape(64,6,5,1,256,256) # Add channel dimension
        # size (64, 6, 5, 3)
        targets = torch.stack(torch.meshgrid(*[torch.arange(i) for i in images.size()[:-3]],indexing='ij'),dim=-1)
        dataset = TensorDataset(images,targets,**kwargs)
        
    elif dataset_name == 'hvm':
        images = torch.from_numpy(np.load(os.path.join(env_config['hvm_path'], 'HVM_images.npy'))[640:])
        images = [T.ToPILImage()(image).convert("RGB") for image in images]
        
        categories = [class_name.decode('utf-8') for class_name in np.load(os.path.join(env_config['hvm_path'], 'HVM_obj_categories.npy'))]
        classes = [class_name.decode('utf-8') for class_name in np.load(os.path.join(env_config['hvm_path'], 'HVM_obj_classes.npy'))]
        category_to_idx = np.load(os.path.join(env_config['hvm_path'], 'HVM_obj_category_to_idx.npy'), allow_pickle=True)[()]
        class_to_idx = np.load(os.path.join(env_config['hvm_path'], 'HVM_obj_class_to_idx.npy'), allow_pickle=True)[()]
        category_indices = [category_to_idx[class_name] for class_name in categories]
        class_indices = [class_to_idx[class_name] for class_name in classes]
        
        targets = []
        target_types = ['positions', 'sizes', 'poses']
        for target_type in target_types:
            targets.append(np.load(os.path.join(env_config['hvm_path'], f'HVM_obj_{target_type}.npy')))
        targets = list(zip(category_indices,class_indices,*targets))
        
        dataset = ListDataset(images,targets,**kwargs) # image, (category, class, position, size, pose)
        dataset.categories = list(set(categories))
        dataset.classes = list(set(classes))
        dataset.category_to_idx = category_to_idx
        dataset.class_to_idx = class_to_idx
        
    elif dataset_name == 'hk2':
        images = torch.from_numpy(np.load(os.path.join(env_config['hk2_path'], 'HK2_images.npy')))
        images = [T.ToPILImage()(image) for image in images.permute(0,3,1,2)]
        classes = np.load(os.path.join(env_config['hk2_path'], 'HK2_classes.npy'))
        class_to_idx = np.load(os.path.join(env_config['hk2_path'], 'HK2_class_to_idx.npy'), allow_pickle=True)[()]
        targets = [class_to_idx[str(class_name)] for class_name in classes] # stand-in for true labels
        
        dataset = ListDataset(images,targets,**kwargs)
        dataset.classes = classes
        dataset.class_to_idx = class_to_idx
        
    elif dataset_name == 'rust':
        images = torch.from_numpy(np.load(os.path.join(env_config['rust_path'], 'Rust_images.npy')))
        images = [T.ToPILImage()(images[i]).convert("RGB") for i in range(images.shape[0])]
        targets = np.arange(len(images))
        dataset = ListDataset(images,targets,**kwargs)

    elif dataset_name == 'mkturk_test':
        images = torch.from_numpy(np.load(os.path.join(env_config['mkturk_path'], 'test/images.npy')))
        targets = torch.stack(torch.meshgrid(*[torch.arange(i) for i in images.size()[:-3]],indexing='ij'),dim=-1)
        dataset = TensorDataset(images,targets,**kwargs)
        
    else:
        raise NotImplementedError(f"dataset_name {dataset_name} not implemented. Try 'imagenet', 'pseudo_hvm', 'hvm', 'hk2', or 'rust'.")
    
    return dataset

def _to_tensor(lst):
    return [torch.from_numpy(elem) if isinstance(elem, np.ndarray) else torch.Tensor([elem]) for elem in lst]
        
class AbstractStructuredDataset(ABC, torch.utils.data.Dataset):
    def __init__(self, return_index):
        self.return_index = return_index
        
    @property
    @abstractmethod
    def shape(self):
        pass
        
    def _out(self, image, target, index):
        if self.return_index:
            return image, target, torch.Tensor(list(np.unravel_index(index, self.shape)))
        return image, target
    
    def __len__(self):
        return utils.prod(self.shape)
    
### Classes for constructing an AbstractStructuredDataset ###
    
class StructuredDataset(AbstractStructuredDataset):
    """
    dataset - a torch.utils.data.Dataset instance
    """
    def __init__(self, dataset, return_index=False):
        super().__init__(return_index)
        self.dataset = dataset
        
    def shape(self):
        return (len(self.dataset),)
        
    def __getitem__(self, index):
        image, target = self.dataset[index]
        return super()._out(image, target, index)
    
class ListDataset(AbstractStructuredDataset):
    def __init__(self, images, targets, transform=None, target_transform=None, return_index=False):
        assert len(images) == len(targets)
        super().__init__(return_index)
        self.images = images
        self.targets = [_to_tensor(target) if isinstance(target, list) or isinstance(target, tuple) else _to_tensor([target]) for target in targets]
        self.transform = transform
        self.target_transform = target_transform

    @property
    def shape(self):
        return (len(self.images),)
    
    def __getitem__(self, index):
        image, target = self.images[index], self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return super()._out(image, target, index)
    
class TensorDataset(AbstractStructuredDataset):
    """
    A dataset which can be manipulated like a tensor. Must set is_tensor to False before using this dataset in a dataloader.
    """
    def __init__(self, images, targets, transform=None, target_transform=None, return_index=False):
        # Assumes images is a tensor of size (...,C,H,W), targets is a tensor of size(...,M)
        assert images.size()[:-3] == targets.size()[:-1]
        super().__init__(return_index)
        self._shape, self._image_shape, self._target_shape = images.size()[:-3], images.size()[-3:], targets.size()[-1:]
        self.images = images
        self.targets = targets 
        self.transform = transform
        self.target_transform = target_transform
        self._is_tensor = True
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def is_tensor(self):
        return self._is_tensor
    
    @is_tensor.setter
    def is_tensor(self, is_tensor):
        if self._is_tensor and not is_tensor:
            # Flattens tensors and convert images to PIL format
            self.images = [T.ToPILImage()(image).convert("RGB") for image in self.images.reshape(-1,*self._image_shape)]
            self.targets = self.targets.reshape(-1,*self._target_shape)
        elif not self._is_tensor and is_tensor:
            raise NotImplementedError("")
        else:
            pass
        self._is_tensor = is_tensor
    
    def __getitem__(self, index):
        if self._is_tensor:
            images, targets = self.images[index], self.targets[index]
            return TensorDataset(images, targets, transform=self.transform, target_transform=self.target_transform, return_index=self.return_index)
        else:
            image, target = self.images[index], self.targets[index]
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return super()._out(image, target, index)
        
    def permute(self, *args):
        assert self._is_tensor # This method is only accessible to a tensor-like dataset
        images = self.images.permute(*args, *[i + len(self._shape) for i in range(len(self._image_shape))])
        targets = self.targets.permute(*args, *[i + len(self._shape) for i in range(len(self._target_shape))])
        return TensorDataset(images, targets, transform=self.transform, target_transform=self.target_transform, return_index=self.return_index)
    
    def reshape(self, *args):
        assert self._is_tensor # This method is only accessible to a tensor-like dataset
        images, targets = self.images.reshape(*args, *self._image_shape), self.targets.reshape(*args, *self._target_shape)
        return TensorDataset(images, targets, transform=self.transform, target_transform=self.target_transform, return_index=self.return_index)
    
### Classes for augmenting an AbstractStructuredDataset ###
    
class OuterDataset(AbstractStructuredDataset):
    """
    dataset - a AbstractStructuredDataset instance
    """
    def __init__(self, dataset, outer_shape, return_index=False):
        super().__init__(return_index)
        self.dataset = dataset
        self.dataset.return_index = False # No need for self.dataset to return_index
        self.outer_shape = outer_shape
        
    @property
    def shape(self):
        return (*self.dataset.shape,*self.outer_shape)
        
    def __getitem__(self, index):
        image, target = self.dataset[index // utils.prod(self.outer_shape)]
        return super()._out(image, target, index)
    
class RandomTransformDataset(AbstractStructuredDataset):
    """
    A dataset together with custom random transforms
    whose data are the transformed images and the labels are the transform values 
    
    Inputs:
    dataset - a AbstractStructuredDataset instance
    transform - a LabeledTransform instance
    """
    def __init__(self, dataset, transform, identity=False, target_transform=None, return_index=False):
        super().__init__(return_index)
        self.dataset = dataset
        self.dataset.return_index = False # No need for self.dataset to return_index
        self.transform = transform
        self.identity = identity
        self.target_transform = target_transform
        
    @property
    def shape(self):
        return self.dataset.shape

    def __getitem__(self, index):
        image, identity = self.dataset[index]
        params = self.transform.get_params(image)
        target = self.transform.get_target(image, params)
        image = self.transform(image, params) # transform images
        if self.identity:
            if isinstance(identity, list):
                target = [*identity, *target]
            else:
                target = [identity, *target]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return super()._out(image, target, index)
