import argparse
import os
from importlib import import_module

import yaml
with open(os.path.join("custom","test","config.yaml"), "r") as f:
	config = yaml.load(f, Loader=yaml.FullLoader)

recipe = getattr(import_module("custom.test.recipe"),"Recipe")(config)

model = recipe.getModel()
trainDataloader, validDataloader = recipe.getDataloader()
optimizer = recipe.getOptimizer()
criterion = recipe.getLoss()

from src.trainer import Trainer

t = Trainer(config, trainDataloader, validDataloader, model, optimizer,criterion,None)
t.start()