from client.infercode_trainer import InferCodeTrainer
import configparser
import logging
logging.basicConfig(level=logging.INFO)

config = configparser.ConfigParser()
config.read("configs/java_small_config.ini")

infercode_trainer = InferCodeTrainer(config)
infercode_trainer.train()