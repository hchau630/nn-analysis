import argparse
import uuid
import pickle
import pathlib

from nn_analysis import metrics as me
from nn_analysis import utils
from nn_analysis.constants import ENV_CONFIG_PATH, METRIC_CONFIGS_PATH, MODEL_CONFIGS_PATH, ARCH_CONFIGS_PATH

env_config = utils.load_config(ENV_CONFIG_PATH)
metric_configs = utils.load_config(METRIC_CONFIGS_PATH)
model_configs = utils.load_config(MODEL_CONFIGS_PATH)
arch_configs = utils.load_config(ARCH_CONFIGS_PATH)
    
def main(model_name, layers, metric_name, version, epoch=None, overwrite=False, debug=False, verbose=False):
    metric_config = metric_configs[metric_name][f"{version:02d}"]
    metric = me.custom_metrics.metrics_dict[metric_config['class']](**metric_config['kwargs'])
    layer_names = arch_configs[model_configs[model_name]['arch']]['layer_names']
    
    for layer in layers:
        layer_name = layer_names[layer]
        
        if me.utils.data_exists(model_name, epoch, layer_name, metric_name, version):
            print(f"Data for layer {layer} exists. Skipping this layer.")
            continue
            
        result = metric.evaluate(model_name, epoch, layer_name)
        
        print(f"Done layer {layer}, saving results...")
        
        if not debug:
            me.utils.save_data(model_name, epoch, layer_name, metric_name, version, result, overwrite=overwrite)
            
    print("Done")

if __name__ == '__main__':
    print("Started main_analyze.py...")
    parser = argparse.ArgumentParser(description='Analyze activations')
    parser.add_argument('model_name', type=str,
                        help='specify the model of which activations are saved')
    parser.add_argument('layers', type=int, nargs='+',
                        help='specify the layers of which activations are saved')
    parser.add_argument('metric_name', type=str,
                        help='specify what activations are saved')
    parser.add_argument('version', type=int,
                        help='Version number. Error will be raised if the result for the specified version' \
                             'already exists and --overwrite flag is not provided.')
    parser.add_argument('--epoch', type=int,
                        help='specify the epoch of the model (default: None)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allows overwriting specified version.')
    parser.add_argument('--debug', action='store_true',
                        help='use debug mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='verbose output')
    args = parser.parse_args()
    main(args.model_name, args.layers, args.metric_name, args.version, epoch=args.epoch, overwrite=args.overwrite, debug=args.debug, verbose=args.verbose)