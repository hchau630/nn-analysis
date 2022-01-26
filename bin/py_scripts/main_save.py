import argparse
import uuid

from nn_analysis import utils
from nn_analysis import acts as ac
from nn_analysis.constants import MODEL_CONFIGS_PATH, ARCH_CONFIGS_PATH, ACTS_CONFIGS_PATH, ENV_CONFIG_PATH

model_configs = utils.load_config(MODEL_CONFIGS_PATH)
arch_configs = utils.load_config(ARCH_CONFIGS_PATH)
acts_configs = utils.load_config(ACTS_CONFIGS_PATH)
env_config = utils.load_config(ENV_CONFIG_PATH)

def main(model_name, layers, acts_name, version, epoch=None, overwrite=False, debug=False, verbose=False):
    arch = model_configs[model_name]['arch']
    layer_names = [arch_configs[arch]['layer_names'][layer] for layer in layers]
    acts_config = acts_configs[acts_name][f"{version:02d}"]
    
    filtered_layers, filtered_layer_names = [], []
    for layer, layer_name in zip(layers, layer_names):
        if not ac.utils.data_exists(model_name, epoch, acts_name, version, layer_name=layer_name, data_type='y'):
            filtered_layers.append(layer)
            filtered_layer_names.append(layer_name)
    layers, layer_names = filtered_layers, filtered_layer_names
    
    print(f"Layers after filtering: {layers}")
    if len(layers) == 0:
        print("Number of layers is 0. Exiting")
        return
    
    # Configure tmp_acts_file
    tmp_acts_file = f"{env_config['tmp_path']}/acts/{uuid.uuid4()}.hdf5" # generate random file name so that multiple programs can be run in parallel. If there's a clash, well, you're the unluckiest person to have ever existed.
    
    save_kwargs = {k: v for k, v in acts_config['kwargs'].items() if k in ['save_loc']}
    
    acts = ac.Acts(model_name, epoch, acts_config['dataset_configs'])
    
    with utils.TmpFileHandler(tmp_acts_file): # removes tmp_acts_file upon encountering exception or SIGTERM
        kwargs = {k: v for k, v in acts_config['kwargs'].items() if k in ['n_samples', 'batch_size']}
        
        acts.save_tmp_acts(tmp_acts_file, layer_names, **kwargs, verbose=verbose)

        kwargs = {k: v for k, v in acts_config['kwargs'].items() if k in ['transform']}
        data_dict = acts.get_x(tmp_acts_file, **kwargs)
        
        if not debug:
            ac.utils.save_data(model_name, epoch, acts_name, version, data_dict, overwrite=overwrite, **save_kwargs)

        for layer, layer_name in zip(layers, layer_names):
            kwargs = {k: v for k, v in acts_config['kwargs'].items() if k in ['transform', 'n_pcs', 'svd_solver']}
            data_dict = acts.get_y(tmp_acts_file, layer_name, **kwargs)
            
            print(f"Layer: {layer}. Explained variance ratio: {data_dict['evr'][-1] if 'evr' in data_dict else None}")
                
            if not debug:
                ac.utils.save_data(model_name, epoch, acts_name, version, data_dict, layer_name=layer_name, overwrite=overwrite, **save_kwargs)
                
    print("Done.")
    
if __name__ == '__main__':
    print("Started main_save.py...")
    parser = argparse.ArgumentParser(description='Save activations')
    parser.add_argument('model_name', type=str,
                        help='Specify the model of which activations are saved')
    parser.add_argument('layers', type=int, nargs='+',
                        help='Specify the layers of which activations are saved')
    parser.add_argument('acts_name', type=str,
                        help='Specify what activations are saved')
    parser.add_argument('version', type=int,
                        help='Version number. Error will be raised if the activations for the specified version' \
                             'already exists and --overwrite flag is not provided.')
    parser.add_argument('--epoch', type=int,
                        help='Specify the epoch of the model (default: None)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allows overwriting specified version.')
    parser.add_argument('--debug', action='store_true',
                        help='Use debug mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()
    main(args.model_name, args.layers, args.acts_name, args.version, epoch=args.epoch, overwrite=args.overwrite, debug=args.debug, verbose=args.verbose)