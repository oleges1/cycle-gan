# from time import sleep
# sleep(21600)

from train import *
import yaml

with open('configs/edges2shoes_paired_with_D.yaml', 'r') as f:
    config = DotDict(yaml.safe_load(f))
    
train(config)