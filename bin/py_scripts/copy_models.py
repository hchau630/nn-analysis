import argparse
import sys
import pathlib

from nn_analysis.constants import ENV_CONFIG_PATH, MODEL_CONFIGS_PATH
from nn_analysis import utils

env_config = utils.load_config(ENV_CONFIG_PATH)
model_configs = utils.load_config(MODEL_CONFIGS_PATH)

def main(model_names, dest):
    for model_name in model_names:
        source_path = f"{env_config['model_base_path']}/{model_configs[model_name]['path']}"
        dest_path = f"{dest}/{model_configs[model_name]['path']}"
        print(f"{source_path},{pathlib.Path(dest_path).parent}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_names', type=str, nargs='+', 
                        help='specify the models of which activations are saved')
    parser.add_argument('dest', type=str,
                        help='destination directory')
    args = parser.parse_args()
    main(args.model_names, args.dest)