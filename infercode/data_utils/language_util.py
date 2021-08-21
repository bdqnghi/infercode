from .vocabulary import Vocabulary
from tree_sitter import Language, Parser
import numpy as np
from bidict import bidict

class LanguageUtil():

    def __init__(self):
        self.languages = bidict({
                "java": ".java", 
                "c": ".c", 
                "c_sharp": ".cs", 
                "cpp": ".cpp",
                "bash": ".sh",
                "go": ".go",
                "javascript": ".js",
                "lua": ".lua",
                "php": ".php",
                "python": ".py",
                "ruby": ".rb",
                "rust": ".rs",
                "scala": ".scala",
                "kotlin": ".kt",
                "solidity": ".sol",
                "html": ".html",
                "css": ".css",
                "haskell": ".hs",
                "r": ".r"
        })
    
    def get_language_by_file_extension(self, extension):
        return self.languages.inverse[extension]
        
    def get_language_index(self, language):
        return list(self.languages.keys()).index(language)

    def get_num_languages(self):
        return len(self.languages.keys())