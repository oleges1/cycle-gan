from train import *
import yaml

with open('configs/face2simpson_unpaired.yaml', 'r') as f:
    config = DotDict(yaml.safe_load(f))
    
train(config)