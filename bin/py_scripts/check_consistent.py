from nn_analysis.constants import ACTS_CONFIGS_PATH
from nn_analysis import utils
from nn_analysis import acts as ac

def main():
    acts_configs = utils.load_config(ACTS_CONFIGS_PATH)
    
    for acts_name in acts_configs.keys():
        for version in acts_configs[acts_name].keys():
            version = int(version)
            ac.utils.assert_consistent_x(acts_name, version)

if __name__ == '__main__':
    main()