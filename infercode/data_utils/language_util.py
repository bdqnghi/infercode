import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from os import path
from .vocabulary import Vocabulary
from tree_sitter import Language, Parser
from pathlib import Path
import glob, os
import numpy as np


class LanguageUtil():

    def __init__(self):
        self.languages = [
                "java", 
                "c", 
                "c_sharp", 
                "cpp", 
                "bash", 
                "go",
                "javascript",
                "lua",
                "php",
                "python",
                "ruby",
                "rust",
                "r",
                "scala",
                "haskell",
                "lua",
                "kotlin",
                "solidity"
                "bash",
                "html",
                "css",
                "markdown"
        ]

    def get_language_index(self, language):
        return self.languages.index(language)

    def get_num_languages(self):
        return len(self.languages)