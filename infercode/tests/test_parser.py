import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from os import path
from tree_sitter import Language, Parser
from pathlib import Path
import glob, os

home = str(Path.home())
cd = os.getcwd()
os.chdir(path.join(home, ".tree-sitter", "bin"))
Languages = {}
for file in glob.glob("*.so"):
  try:
    lang = os.path.splitext(file)[0]
    Languages[lang] = Language(path.join(home, ".tree-sitter", "bin", file), lang)
  except:
    print("An exception occurred to {}".format(lang))
os.chdir(cd)
parser = Parser()
lang = Languages.get("bash")
parser.set_language(lang)


ast = parser.parse(open("test.sh", "rb").read())
print(ast.root_node.sexp())
print(ast.root_node.children[0])