from client.infercode_trainer import InferCodeTrainer
import configparser
import logging

logging.basicConfig()
config = configparser.ConfigParser()
config.read("training_config.ini")

infercode_trainer = InferCodeTrainer(config)
infercode_trainer.train()