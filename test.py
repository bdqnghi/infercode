import os
import logging
logging.basicConfig(level=logging.INFO)
# Change from -1 to 0 to enable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from infercode.client.infercode_client import InferCodeClient
i = InferCodeClient(language="c")
i.init_from_config()
vectors = i.encode(["for (i = 0; i < n; i++)", "struct book{ int num; char s[27]; }shu[1000];"])

print(vectors)

