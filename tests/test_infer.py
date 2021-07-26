import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from infercode.client.infercode_client import InferCodeClient
import logging
logging.basicConfig(level=logging.INFO)

# import configparser 
# config = configparser.ConfigParser()
# config.read("../configs/OJ_raw_small.ini")

infercode = InferCodeClient(language="c")
infercode.init_from_config()
vectors = infercode.encode(["for (i = 0; i < n; i++)", "struct book{ int num; char s[27]; }shu[1000];"])

print(vectors)
# vectors = infercode.encode(["for int i = 0"])


# print(vectors )