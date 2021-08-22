from setuptools import setup, find_packages


install_requires=[
    "bidict==0.21.2",
    "coloredlogs == 15.0.1",
    "dpu_utils==0.2.19",
    "keras_radam==0.15.0",
    "numpy==1.19.4",
    "protobuf==3.14.0",
    "scikit_learn==0.24.0",
    "scipy==1.5.2",
    "sentencepiece==0.1.95",
    "tensorflow==2.4.0",
    "tqdm==4.55.1",
    "tree_sitter==0.19.0",
    "utils==1.0.1"
]

setup(
  name = 'infercode',
  version = '0.0.18',
  py_modules = ['infercode'],
  description = 'Map any code snippet into vector',
  author = 'Nghi D. Q. Bui and Yijun Yu',
  author_email = 'bdqnghi@gmail.com',
  license="MIT",
  url = 'https://github.com/bdqnghi/infercode/',
  classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Intended Audience :: Developers',
  ],
  package_dir={"infercode": "infercode"},
  packages=find_packages(where=".", exclude=["tests", "logo", "datasets"]),
  package_data={'infercode': ['configs/*.ini', 'sentencepiece_vocab/*', 'sentencepiece_vocab/node_types/*' , 'sentencepiece_vocab/subtrees/*', 'sentencepiece_vocab/tokens/*' ]},
  install_requires=install_requires,
  include_package_data=True,
)
