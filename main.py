from train import *
import yaml

with open('configs/imdb2simpson_unpaired.yaml', 'r') as f:
    config = DotDict(yaml.safe_load(f))
    
train(config)