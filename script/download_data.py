import urllib.request
import zipfile
from tqdm import tqdm
import os

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)\


code_classification_data_url = "https://ai4code.s3-ap-southeast-1.amazonaws.com/OJ_pycparser_train_test_val.zip"
code_classification_output_path = "../OJ_pycparser_train_test_val.zip"

pretrained_model_url = "https://ai4code.s3-ap-southeast-1.amazonaws.com/model_OJ_pycparser.zip"
pretrained_model_output_path = "../model_OJ_pycparser.zip"

if not os.path.exists(code_classification_output_path):
    download_url(code_classification_data_url, code_classification_output_path)

if not os.path.exists(pretrained_model_output_path):
    download_url(pretrained_model_url, pretrained_model_output_path)

with zipfile.ZipFile(code_classification_output_path) as zf:
    for member in tqdm(zf.infolist(), desc='Extracting '):
        try:
            zf.extract(member, "../")
        except zipfile.error as e:
            pass


with zipfile.ZipFile(pretrained_model_output_path) as zf:
    for member in tqdm(zf.infolist(), desc='Extracting '):
        try:
            zf.extract(member, "../")
        except zipfile.error as e:
            pass