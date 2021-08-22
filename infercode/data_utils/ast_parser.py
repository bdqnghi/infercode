import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from os import path
from .vocabulary import Vocabulary
from .language_util import LanguageUtil
from tree_sitter import Language, Parser
from pathlib import Path
import glob, os
import numpy as np
import logging
import urllib.request
from tqdm import tqdm
import zipfile
import shutil

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

class ASTParser():
    import logging
    LOGGER = logging.getLogger('ASTParser')
    def __init__(self, language=None):


        # ------------ To initialize for the treesitter parser ------------
        home = str(Path.home())
        cd = os.getcwd()
        p = path.join(home, ".tree-sitter")
        if not path.exists(p):
                os.makedirs(p, exist_ok=True)
                zip_url = "https://github.com/yijunyu/tree-sitter-parsers/archive/refs/heads/main.zip"
                parsers_target = os.path.join(p, "main.zip")
                download_url(zip_url, parsers_target)
                with zipfile.ZipFile(parsers_target, 'r') as zip_ref:
              	        zip_ref.extractall(p)
              	        shutil.move(path.join(p, "tree-sitter-parsers-main"), path.join(p, "bin"))
              	        os.remove(parsers_target)
        p = path.join(p, "bin")
        os.chdir(p)
        self.Languages = {}
        for file in glob.glob("*.so"):
          try:
            lang = os.path.splitext(file)[0]
            self.Languages[lang] = Language(path.join(p, file), lang)
          except:
            print("An exception occurred to {}".format(lang))
        os.chdir(cd)
        self.parser = Parser()
        
        self.language = language
        if self.language == None:
            self.LOGGER.info("Cannot find language configuration, using java parser to parse the code into AST")
            self.language = "java"

        lang = self.Languages.get(self.language)
        self.parser.set_language(lang)
        # -----------------------------------------------------------------

       
    def parse_with_language(self, code_snippet, language):
        lang = self.Languages.get(language)
        self.parser.set_language(lang)
        return self.parser.parse(code_snippet)

    def parse(self, code_snippet):
        return self.parser.parse(code_snippet)
    
    def set_language(self, language):
        lang = self.Languages.get(language)
        self.parser.set_language(lang)
