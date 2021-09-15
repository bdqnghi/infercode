import tree_sitter_parsers
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from infercode.client.infercode_client import InferCodeClient
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('InferCodeModel').propagate = False
logging.getLogger('InferCodeTrainer').propagate = False
# Change from -1 to 0 to enable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

language = {
    ".c": "c",
    ".cs": "c_sharp",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".css": "css",
    ".elm": "elm",
    ".go": "go",
    ".html": "html",
    ".hs": "haskell",
    ".java": "java",
    ".js": "javascript",
    ".kt": "kotlin",
    ".lua": "lua",
    ".php": "php",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".sol": "solidity",
    ".sh": "bash",
    ".v": "verilog",
    ".yaml": "yaml",
    ".yml": "yaml",
}

def main(): 
    parser = argparse.ArgumentParser(usage='infercode code')
    parser.add_argument('files', metavar='C', nargs='+', help='a file for the conversion')
    args = parser.parse_args()
    for file in args.files:
        filename, file_extension = os.path.splitext(file)
        infercode = InferCodeClient(language=language[file_extension])
        infercode.init_from_config()
        with open (file, "r") as myfile:
            code=myfile.read()
            logging.getLogger('tensorflow').propagate = False
            vectors = infercode.encode([code])
            print(vectors)

if __name__ == '__main__':
    main()
