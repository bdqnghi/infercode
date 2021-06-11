import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
import random
import pickle
from utils.data.tree_processor import TreeProcessor
import argument_parser
# import utils.network.treecaps_2 as network
import os
import re
import time
from bidict import bidict
import copy
import numpy as np
from utils import evaluation



def main(opt):
    tree_processor = TreeProcessor(opt)
    tree_processor.process_data()

if __name__ == "__main__":
    opt = argument_parser.parse_arguments()
    main(opt)