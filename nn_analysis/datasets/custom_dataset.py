import torch
import torchvision.transforms as T
import numpy as np

from nn_analysis.datasets import datasets as ds
from nn_analysis.datasets import transforms

def get_custom_dataset(base_dataset_name, seed, transform_names=[], subset_indices=None, outer_dims=None):
    transforms_map = {
        'crop': transforms.RandomResizedCrop(224,scale=(0.2,1.),ratio=(1.0,1.0)),
        'color': transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    }
    transforms_list = [transforms_map[transform_name] for transform_name in transform_names]
    transforms_list += [
        T.Resize(224),
        T.ToTensor(),
        transforms.Normalize('imagenet'),
    ]
    
    base_dataset = ds.get_dataset(base_dataset_name)
    
    if subset_indices is not None:
        subset_indices = tuple([slice(*nth_idx) if isinstance(nth_idx, list) else nth_idx for nth_idx in subset_indices])
        if isinstance(base_dataset, ds.TensorDataset):
            base_dataset = base_dataset[subset_indices]
        elif isinstance(base_dataset, ds.ListDataset):
            assert len(subset_indices) == 1 # 1 dimensional dataset
            base_dataset = ds.ListDataset(
                [base_dataset.images[idx] for idx in np.arange(base_dataset.shape[0])[subset_indices]],
                [base_dataset.targets[idx] for idx in np.arange(base_dataset.shape[0])[subset_indices]]
            )
        else:
            raise NotImplementedError("The configuration 'subset_indices' is not implemented for a base_dataset that is neither a TensorDataset nor a ListDataset. Set 'subset_indices' to null if you don't want this functionality.")
    
    if isinstance(base_dataset, ds.TensorDataset):
        base_dataset.is_tensor = False
    
    if outer_dims is not None:
        base_dataset = ds.OuterDataset(base_dataset, outer_dims)
    
    dataset = ds.RandomTransformDataset(
        base_dataset,
        transforms.Compose(transforms_list),
        target_transform=lambda target: torch.cat(target),
        identity=True,
        return_index=True,
    )
    
    return dataset