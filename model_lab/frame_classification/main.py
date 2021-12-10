import argparse
import os
from importlib import import_module
from src.set_seed import setSeed
import yaml



RECIPE = "hospital"

with open(os.path.join("custom",RECIPE,"config.yaml"), "r") as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
config["custom_name"] = RECIPE
setSeed(config["seed"])
recipe = getattr(import_module(f"custom.{RECIPE}.recipe"),"Recipe")(config)

model = recipe.getModel()
trainDataloader, validDataloader = recipe.getDataloader()
optimizer = recipe.getOptimizer()
criterion = recipe.getLoss()
scheduler = recipe.getScheduler()

from src.trainer import Trainer

t = Trainer(config, trainDataloader, validDataloader, model, optimizer,criterion,scheduler)
t.start()