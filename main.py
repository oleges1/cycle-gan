from train import *
import yaml
import os

os.system('bash download.sh')

for config_name in os.listdir('configs'):
    with open(os.path.join('configs', config_name), 'r') as f:
        config = DotDict(yaml.safe_load(f))
    train(config)