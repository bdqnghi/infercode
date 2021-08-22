import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from .vocabulary import Vocabulary
from pathlib import Path
import os
from tqdm import *
from .subtree_util import SubtreeUtil
from .ast_parser import ASTParser
from .language_util import LanguageUtil
import os
import queue
import threading
import random

class TimeLimitExpired(Exception): pass

class SubtreeVocabExtractor():


    def __init__(self, subtree_vocab_model_prefix: str):

        self.subtree_vocab_model_prefix = subtree_vocab_model_prefix
        self.subtree_vocab = Vocabulary(1000000)
        self.subtree_util = SubtreeUtil()
        self.language_util = LanguageUtil()
        # self.ast_util = ASTUtil(node_type_vocab_model_path=node_type_vocab_model_path, 
        #                         node_token_vocab_model_path=node_token_vocab_model_path, language=language)
        self.temp_subtrees_file = "temp_subtrees.csv"
        if os.path.exists(self.temp_subtrees_file):
            os.remove(self.temp_subtrees_file)

    def detect_language_of_file(self, file_path: str):
        _, file_extension = os.path.splitext(file_path)
        return self.language_util.get_language_by_file_extension(file_extension)

    def create_vocab_from_dir(self, input_data_path: str):

        pathqueue = queue.Queue()
        resultqueue = queue.Queue()
        for i in range(0, os.cpu_count()):
            subtree_thread = SubtreeProcessThread(pathqueue, resultqueue)
            subtree_thread.setDaemon(True)
            subtree_thread.start()
          

        write_thread = WriteThread(resultqueue, self.temp_subtrees_file)
        write_thread.setDaemon(True)
        write_thread.start()
     
        all_file_paths = []
        for subdir , dirs, files in os.walk(input_data_path): 
            for file in tqdm(files):
                file_path = os.path.join(subdir, file)
                with open(file_path, "rb") as f:
                    code_snippet = f.read()

                language = self.detect_language_of_file(file_path)
                # tree = self.ast_parser.parse(code_snippet, language)
                # subtrees = self.subtree_util.extract_subtrees(tree)
                pathqueue.put((code_snippet, language))
            
        pathqueue.join()
        resultqueue.join()

        # all_subtrees_vocab = []
        with open(self.temp_subtrees_file, "r") as f1:
            all_subtrees_vocab = f1.read().splitlines()

        all_subtrees_vocab = list(set(all_subtrees_vocab))

        # model_type must be "word" for subtree vocab
        self.subtree_vocab.create_vocabulary(tokens=all_subtrees_vocab, 
                                            model_filename=self.subtree_vocab_model_prefix, 
                                            model_type="word") 
        return self.subtree_vocab
        

class SubtreeProcessThread(threading.Thread):
    def __init__(self, in_queue, out_queue):
        threading.Thread.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.subtree_util = SubtreeUtil()
        self.ast_parser = ASTParser()

    def run(self):
        while True:
            code_snippet, language = self.in_queue.get()
            result = self.process(code_snippet, language)
            self.out_queue.put(result)
            self.in_queue.task_done()

    def _stop(self):
        if self.isAlive():
            Thread._Thread__stop(self)

    def process(self, code_snippet, language):
        # Do the processing job here
        tree = self.ast_parser.parse_with_language(code_snippet, language)
        subtrees = self.subtree_util.extract_subtrees(tree)
        return subtrees

class WriteThread(threading.Thread):
    def __init__(self, queue, output_path):
        threading.Thread.__init__(self)
        self.queue = queue
        self.output_path = output_path

    def write_subtree(self, subtrees):
        for s in subtrees:
            if len(s) > 1 and len(s) < 8:
                # Concat the list of nodes in a subtree into a string
                subtree_str = "-".join(s)
                # if subtree_str not in all_subtrees_vocab:
                # Write to a temporary file as keeping a large array may cause memory overflow
                with open(self.output_path, "a") as f:
                    # all_subtrees_vocab.append(subtree_str)
                    f.write(subtree_str)
                    f.write("\n")
    def run(self):
        while True:
            result = self.queue.get()
            self.write_subtree(result)
            self.queue.task_done()