import os
import time

import torch
import torchvision.transforms as T
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import clear_output
from IPython.display import HTML
import PIL

from nn_analysis.datasets import transforms

def load_img(filename):
    image = PIL.Image.open(filename)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def to_array(image, dtype='torch'):
    """
    Takes in PIL images and converts them to dtype ('torch' or 'np')
    """
    if dtype == 'torch':
        func = T.ToTensor()
    elif dtype == 'np':
        func = np.array
    else:
        raise NotImplementedError("Only dtype 'torch' or 'np' supported")
    
    return func(image)

def to_plot(x, batch=False):
    """
    Takes in a torch or numpy image array and converts it to a format for plotting with plt.imshow.
    If batch=True, assumes the first dimension to be a batch dimension
    """
    image_ndim = x.ndim - 1 if batch else x.ndim
    if image_ndim == 2:
        return x
    elif image_ndim == 3:
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise NotImplementedError("Only torch or numpy image arrays supported")
        
        if batch:
            return np.transpose(x, (0,2,3,1))
        return np.transpose(x, (1,2,0))
    else:
        raise RuntimeError("x must have either 2 (HxW) or 3 (CxHxW) dimensions if batch=False and " + 
                           "either 3 (BxHxW) or 4 (BxCxHxW) dimensions if batch=True")

def grid_imshow(images, normalization=None, ax=None, show=True, nrow=8, **kwargs):
    """
    images is of size BxCxHxW (for colored images) or BxHxW (for b/w images). Can be either numpy array or torch tensor.
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    if normalization is not None:
        images = transforms.UnNormalize(normalization)(images)
    B, H, W = images.size(0), images.size(-2), images.size(-1)
    if images.ndim == 3:
        images = images.unsqueeze(1)
    big_img = make_grid(images,padding=int(0.05*min(H,W)),nrow=nrow,**kwargs)
    if ax is None:
        fig = plt.figure(figsize=(16,2*(1+B//nrow)))
        ax = fig.gca()
    im = ax.imshow(to_plot(big_img),interpolation='nearest')
    if show:
        plt.show()
    return im

def animate(images, normalization=None):
    """
    Creates an animation out of a list of PIL images or a batched torch/numpy array 
    """
    if isinstance(images, list) and len(images) > 1 and isinstance(images[0], PIL.Image.Image):
        images = [to_array(image, dtype='torch') for image in images]
        images = torch.stack(images, dim=0)
    elif (isinstance(images, np.ndarray) or isinstance(images, torch.Tensor)) and (images.ndim == 3 or images.ndim ==4):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
    else:
        raise NotImplemented("Images must be either a list of PIL images or a batched torch/numpy array")
        
    if normalization is not None:
        images = transforms.UnNormalize(normalization)(images)
        
    images = to_plot(images, batch=True)
    
    frames = [] # for storing the generated images
    fig, ax = plt.subplots()
    for i in range(images.shape[0]):
        frames.append([ax.imshow(images[i],animated=True,interpolation='nearest')])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    return HTML(ani.to_jshtml())

def subplots(n_rows, n_cols, size=(5,4), polar=False, **kwargs):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*size[0], n_rows*size[1]), subplot_kw={'polar': polar}, facecolor='white', **kwargs)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape((1,-1))
    elif n_cols == 1:
        axes = axes.reshape((-1,1))
    else:
        pass
    return fig, axes

def savefig(fig, filename):
    folder = os.path.split(filename)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(filename)
    
def get_colors(N):
    if N <= 10:
        return [f'C{i}' for i in range(N)]
    return plt.cm.jet(np.linspace(0,1,N))

def r_plot(ax, x, y, color=None, label=None, **kwargs):
    N = len(x)
    if N == 0:
        angles, values = [], []
    else:
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles = angles + angles[:1]
        values = list(y)
        values = values + values[:1]
    ax.set_rlabel_position(2.0)
    line1, = ax.plot(angles, values, linewidth=1, linestyle='solid', color=color, label=label, **kwargs)
#     line2, = ax.fill(angles, values, alpha=0.1, color=color, **kwargs)
    return line1,

def r_set_data(lines, x, y):
    N = len(x)
    assert N > 0
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles = angles + angles[:1]
    values = list(y)
    values = values + values[:1]
    for line in lines:
        line.set_data(angles, values)
    
def r_xticks(ax, x, x_offset=0.15, y_offset=0.1, size=11, color="grey", **kwargs):
    N = len(x)
    angles = [n / float(N) * 360 for n in range(N)]
    _, labels = ax.set_thetagrids(angles, x, color=color, size=size)
    for i, label in enumerate(labels):
        x, y = label.get_position()
        label.set_position((x,y-x_offset*np.cos(angles[i]*np.pi/180)**2-y_offset*(np.exp(-np.abs(angles[i]*np.pi/180-3*np.pi/2))+np.exp(-np.abs(angles[i]*np.pi/180-np.pi/2)))**5))

def r_yticks(ax, min=0.0, max=1.0, steps=4):
    ax.set_rgrids(np.linspace(min, max, steps+1)[1:], color="grey", size=9)
    ax.set_ylim((min, max))

def r_legend(ax, loc=(1.0, 0.6), **kwargs):
    ax.legend(loc=loc, **kwargs)
        
# def animate(results, epochs, model_names, layers, across, **kwargs):
#     if across == 'epoch':
#         colors = get_colors(len(model_names))
#         fig, axes = subplots(1, len(layers), height_per_plot=7, width_per_plot=8, polar=True)
#         lines = np.empty((len(layers), len(model_names)), dtype=object)

#         for k, layer in enumerate(layers):
#             for j, model_name in enumerate(model_names):
#                 lines[k,j] = r_plot(axes[0,k], [], [], color=colors[j], label=model_name)
#             r_xticks(axes[0,k], list(results.keys()), **kwargs)
#             r_yticks(axes[0,k])
#             r_legend(axes[0,k], loc=(0.95,0.95))
#             axes[0,k].set_title(f"Layer {layer} metrics")
#         plt.tight_layout()

#         def draw(i):
#             for k, layer in enumerate(layers):
#                 for j, model_name in enumerate(model_names):
#                     r_set_data(lines[k,j], list(results.keys()), [results[key][i,j,k] for key in results.keys()])

#         ani = animation.FuncAnimation(fig, draw, frames=len(epochs))
#         plt.close()
#         return HTML(ani.to_jshtml())
#     elif across == 'layer':
#         colors = get_colors(len(model_names))
#         fig, axes = subplots(1, len(epochs), height_per_plot=7, width_per_plot=8, polar=True)
#         plt.subplots_adjust(left=0.15, right=0.85)
#         lines = np.empty((len(epochs), len(model_names)), dtype=object)

#         for i, epoch in enumerate(epochs):
#             for j, model_name in enumerate(model_names):
#                 lines[i,j] = r_plot(axes[0,i], [], [], color=colors[j], label=model_name)
#             r_xticks(axes[0,i], list(results.keys()), **kwargs)
#             r_yticks(axes[0,i])
#             r_legend(axes[0,i], loc=(0.95,0.95))
#             axes[0,i].set_title(f"Epoch {epoch} metrics")
#         plt.tight_layout()

#         def draw(k):
#             for i, epoch in enumerate(epochs):
#                 for j, model_name in enumerate(model_names):
#                     r_set_data(lines[i,j], list(results.keys()), [results[key][i,j,k] for key in results.keys()])

#         ani = animation.FuncAnimation(fig, draw, frames=len(layers))
#         plt.close()
#         return HTML(ani.to_jshtml())
#     else:
#         raise RuntimeError()
        
# def grid_plot(scores, params_i, params_j, params_k, params_l, names=['model','metric','layer','epoch'], score_name='score', permute=[0,1,2,3], grid=False):
#     # 4D plot
#     params = [params_i, params_j, params_k, params_l]
#     params_i, params_j, params_k, params_l = [params[permute[n]] for n in range(4)]
#     names = [names[permute[n]] for n in range(4)]
#     scores = np.transpose(scores, permute)
    
#     colors = get_colors(len(params_i))
#     fig, axes = subplots(len(params_j), len(params_k), height_per_plot=4, width_per_plot=5)
#     lines = np.empty((len(params_i), len(params_j), len(params_k)), dtype=object)

#     for j, param_j in enumerate(params_j):
#         for k, param_k in enumerate(params_k):
#             for i, param_i in enumerate(params_i):
#                 axes[j,k].plot(params_l, scores[i,j,k,:], color=colors[i], label=param_i)
#             axes[j,k].set_xlabel(names[3])
#             axes[j,k].set_ylabel(score_name)
#             axes[j,k].legend()
#             axes[j,k].set_title(f"{names[1]}: {param_j}, {names[2]}: {param_k}")
#             if grid:
#                 axes[j,k].grid(b=True)
#     plt.tight_layout()
#     plt.show()
        
# def grid_plot_animate(scores, params_i, params_j, params_k, params_l, params_m, names=['model','metric','layer','epoch','n_pcs'], score_name='score', permute=[0,1,2,3,4], plot_func='plot'):
#     # 5D plot lol
#     params = [params_i, params_j, params_k, params_l, params_m]
#     params_i, params_j, params_k, params_l, params_m = [params[permute[n]] for n in range(5)]
#     names = [names[permute[n]] for n in range(5)]
#     scores = np.transpose(scores, permute)
    
#     colors = get_colors(len(params_i))
#     fig, axes = subplots(len(params_j), len(params_k), height_per_plot=4, width_per_plot=5)
#     lines = np.empty((len(params_i), len(params_j), len(params_k)), dtype=object)

#     for j, param_j in enumerate(params_j):
#         for k, param_k in enumerate(params_k):
#             for i, param_i in enumerate(params_i):
#                 if plot_func == 'plot':
#                     lines[i,j,k], = axes[j,k].plot(params_m, scores[i,j,k,0,:], color=colors[i], label=param_i)
#                 elif plot_func == 'scatter':
#                     lines[i,j,k], = axes[j,k].scatter(params_m, scores[i,j,k,0,:], color=colors[i], label=param_i)
#                 else:
#                     raise RuntimeError("plot_func should be 'plot' or 'scatter'")
#             axes[j,k].set_xlabel(names[4])
#             axes[j,k].set_ylabel(score_name)
#             ylim = axes[j,k].get_ylim()
#             axes[j,k].set_ylim((0,ylim[1]))
# #             axes[j,k].set_ylim((0,1))
#             axes[j,k].legend()
#             axes[j,k].set_title(f"{names[1]} {param_j} {names[2]} {param_k}")
#     title = plt.suptitle(f"{names[3]} {params_l[0]}")
#     plt.tight_layout()

#     def draw(l):
#         for i, param_i in enumerate(params_i):
#             for j, param_j in enumerate(params_j):
#                 for k, param_k in enumerate(params_k):
#                     if plot_func == 'plot':
#                         lines[i,j,k].set_data(params_m, scores[i,j,k,l,:])
#                     elif plot_func == 'scatter':
#                         lines[i,j,k].set_offsets(np.stack([params_m, scores[i,j,k,l,:]],axis=0).T)
#                     else:
#                         raise RuntimeError("plot_func should be 'plot' or 'scatter'")
#         title.set_text(f"{names[3]} {params_l[l]}")

#     ani = animation.FuncAnimation(fig, draw, frames=len(params_l))
#     plt.close()
#     return HTML(ani.to_jshtml())