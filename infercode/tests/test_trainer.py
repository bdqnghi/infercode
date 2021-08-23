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
input_data_path = "../../datasets/OJ_raw_small/"
output_processed_data_path = "../../datasets/OJ_raw_processed/OJ_raw_small.pkl"
infercode_trainer = InferCodeTrainer(language="c", 
                                    input_data_path=input_data_path,
                                    output_processed_data_path=output_processed_data_path)
infercode_trainer.init_from_config()
infercode_trainer.train()