import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(ROOT_DIR, 'configs')
ENV_CONFIG_PATH = os.path.join(CONFIGS_DIR, 'env_config.json')
MODEL_CONFIGS_PATH = os.path.join(CONFIGS_DIR, 'model_configs.json')
ARCH_CONFIGS_PATH = os.path.join(CONFIGS_DIR, 'arch_configs.json')
ACTS_CONFIGS_PATH = os.path.join(CONFIGS_DIR, 'acts_configs')
METRIC_CONFIGS_PATH = os.path.join(CONFIGS_DIR, 'metric_configs')