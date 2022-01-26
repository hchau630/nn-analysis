import argparse
import ast
from collections import defaultdict

from nn_analysis import metrics as me
from nn_analysis import acts as ac
from nn_analysis.constants import ENV_CONFIG_PATH, MODEL_CONFIGS_PATH, ARCH_CONFIGS_PATH, METRIC_CONFIGS_PATH
from nn_analysis import utils

env_config = utils.load_config(ENV_CONFIG_PATH)
model_configs = utils.load_config(MODEL_CONFIGS_PATH)
arch_configs = utils.load_config(ARCH_CONFIGS_PATH)
metric_configs = utils.load_config(METRIC_CONFIGS_PATH)

def modify_analyze_config_list_into_save_config_list(config_list):
    lines = []
    for config in config_list:
        # Gather all activation datasets and remove redundancies
        all_acts = []
        for metric_name, version in config['metrics']:
            metric_config = metric_configs[metric_name][f"{version:02d}"]
            metric = me.custom_metrics.metrics_dict[metric_config['class']](**metric_config['kwargs'])
            for acts in metric.required_acts:
                all_acts.append((acts['name'], acts['version']))
        all_acts = list(set(all_acts))

        config['acts'] = all_acts
        del config['metrics']
        
    return config_list

def create_config(filename, config_list, extra_args, config_type):
    d = defaultdict(set)
    
    for config in config_list:        
        # configure epochs
        if 'epochs' in config:
            epochs = list(range(*config['epochs']))
        else:
            epochs = [None]
            
        for model_name in config['model_names']:
            # configure layers
            layer_names = arch_configs[model_configs[model_name]['arch']]['layer_names']
            layers = list(range(len(layer_names)))[slice(*config['layers'])]
            
            for name, version in config[config_type]:
                for epoch in epochs:
                    for layer in layers:
                        layer_name = layer_names[layer]
                        
                        if config_type == 'acts':
                            if ac.utils.data_exists(model_name, epoch, name, version, layer_name=layer_name, data_type='y'):
                                continue
                        elif config_type == 'metrics':
                            if me.utils.data_exists(model_name, epoch, layer_name, name, version):
                                continue
                        else:
                            raise NotImplementedError("config_type must be either 'acts' or 'metrics'.")
                            
                        if epoch is None:
                            key = (model_name, name, version, *extra_args)
                        else:
                            key = (model_name, name, version, '--epoch', epoch, *extra_args)
                            
                        d[key].add(layer)
         
    lines = [(k[0], *sorted(v), *k[1:]) for k, v in d.items()]
    lines = [" ".join([str(elem) for elem in line]) for line in lines]
    
    with open(filename, 'w') as f:
        f.write("\n".join(lines))

def main(config_filename, in_type, out_types, extra_args):
    with open(config_filename, 'r') as f:
        config_list = ast.literal_eval(f.read())
        
    save_config_filename = f"{env_config['script_tmp_path']}/save_config.txt"
    analyze_config_filename = f"{env_config['script_tmp_path']}/analyze_config.txt"
    
    if in_type == 'analyze':
        if 'analyze' in out_types:
            create_config(analyze_config_filename, config_list, extra_args, 'metrics')
            print(f"Created analyze config at: {analyze_config_filename}")
        if 'save' in out_types:
            modify_analyze_config_list_into_save_config_list(config_list)
            create_config(save_config_filename, config_list, extra_args, 'acts')
            print(f"Created save config at: {save_config_filename}")
    elif in_type == 'save':
        if 'analyze' in out_types:
            raise ValueError("out_types cannot contain 'analyze' when in_type is 'save'.")
        if 'save' in out_types:
            create_config(save_config_filename, config_list, extra_args, 'acts')
            print(f"Created save config at: {save_config_filename}")
    else:
        raise NotImplementedError("in_type must be either 'analyze' or 'save'.")
        
if __name__ == '__main__':
    print("Started main_setup.py...")
    parser = argparse.ArgumentParser(description='Tool for setting up the config file for analyze_multi.sh and save_multi.sh.')
    parser.add_argument('config_filename', type=str,
                        help='Name of the .py file that specifcies which models, layers, and acts/metrics to run.')
    parser.add_argument('--in-type', '-i', type=str, choices=['save', 'analyze'], default='analyze',
                        help='Whether the input config file is for saving or analyzing acts. (default: analyze)')
    parser.add_argument('--out-types', '-o', type=str, nargs='+', choices=['save', 'analyze'], default=['save', 'analyze'],
                        help='Whether the output config file is for saving or analyzing acts. Can choose more than one. (default: ["save","analyze"])')
    args, extra_args = parser.parse_known_args()
    main(args.config_filename, args.in_type, args.out_types, extra_args)