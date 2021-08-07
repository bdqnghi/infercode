import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from client.infercode_trainer import InferCodeTrainer
import configparser
import logging
import pathlib
print(pathlib.Path().resolve())
logging.basicConfig(level=logging.INFO)

# config = configparser.ConfigParser()
# config.read("../configs/OJ_raw_small.ini")

infercode_trainer = InferCodeTrainer(language="c")
infercode_trainer.init_from_config()
infercode_trainer.train()