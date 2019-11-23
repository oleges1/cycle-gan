from train import *
import yaml

with open('configs/cityscapes_paired_with_D.yaml', 'r') as f:
    config = DotDict(yaml.safe_load(f))
    
train(config)