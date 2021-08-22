from dpu_utils.codeutils import identifiersplitting
import sys
from tqdm import *
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from .vocabulary import Vocabulary
import os
import queue
import threading

class TokenVocabExtractor():

    def __init__(self, node_token_vocab_model_prefix: str, model_type: str="bpe") -> None:
        self.node_token_vocab_model_prefix = node_token_vocab_model_prefix
        self.model_type = model_type
        self.token_vocab = Vocabulary(1000000)

        self.temp_tokens_file = "temp_tokens.csv"
        if os.path.exists(self.temp_tokens_file):
            os.remove(self.temp_tokens_file)

    def create_vocab_from_dir(self, input_data_path: str):

        pathqueue = queue.Queue()
        resultqueue = queue.Queue()
        for i in range(0, os.cpu_count()):
            subtree_thread = TokenProcessThread(pathqueue, resultqueue)
            subtree_thread.setDaemon(True)
            subtree_thread.start()
          

        write_thread = WriteThread(resultqueue, self.temp_tokens_file)
        write_thread.setDaemon(True)
        write_thread.start()

        for subdir , dirs, files in os.walk(input_data_path): 
            for file in tqdm(files):
                # if file.endswith(file_types):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r", errors='ignore') as f:
                    data = str(f.read())

                    pathqueue.put(data)

        pathqueue.join()
        resultqueue.join()

        self.token_vocab.create_vocabulary_from_file(sp_text_file=self.temp_tokens_file, model_filename=self.node_token_vocab_model_prefix, model_type=self.model_type)        
        return self.token_vocab



class TokenProcessThread(threading.Thread):
    def __init__(self, in_queue, out_queue):
        threading.Thread.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        while True:
            data = self.in_queue.get()
            result = self.process(data)
            self.out_queue.put(result)
            self.in_queue.task_done()

    def _stop(self):
        if self.isAlive():
            Thread._Thread__stop(self)

    def process(self, data):
        # Do the processing job here
        data = data.replace("\n", " ")
        parts = identifiersplitting.split_identifier_into_parts(data)
        return " ".join(parts)

class WriteThread(threading.Thread):
    def __init__(self, queue, output_path):
        threading.Thread.__init__(self)
        self.queue = queue
        self.output_path = output_path

    def write_tokens(self, tokens):
        with open(self.output_path, "a") as f:
            # all_subtrees_vocab.append(subtree_str)
            f.write(tokens)
            f.write("\n")

    def run(self):
        while True:
            result = self.queue.get()
            self.write_tokens(result)
            self.queue.task_done()