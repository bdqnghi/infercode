from setuptools import setup, find_packages

setup(
  name = 'infercode',
  version = '0.0.3',
  py_modules = ['infercode'],
  description = 'Map any code snippet into vector',
  author = 'Nghi D. Q. Bui and Yijun Yu',
  author_email = 'bdqnghi@gmail.com',
  url = 'https://github.com/bdqnghi/infercode/',
  download_url = 'https://github.com/qzb/markdown-jinja/archive/v0.1.0.tar.gz',
  classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Intended Audience :: Developers',
  ],
  install_requires=[
        "bidict==0.21.2",
        "dpu_utils==0.2.19",
        "keras_radam==0.15.0",
        "numpy==1.19.4",
        "protobuf==3.14.0",
        "scikit_learn==0.24.0",
        "scipy==1.5.2",
        "sentencepiece==0.1.95",
        "tensorflow==2.4.0",
        "tqdm==4.55.1",
        "tree_sitter==0.2.1",
        "utils==1.0.1"
    ],
)