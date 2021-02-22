import os
# from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor as WorkerExecutor
import copy
import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--worker", default=6, type=int, help="Num worker")
parser.add_argument("--input_path", default="OJ_raw_train_test_val/val", type=str, help="Input path")
parser.add_argument("--output_path", default="OJ_stmt_train_test_val/val", type=str, help="Output path")

args = parser.parse_args()



def generate_folder_pb(src_path, tgt_path, worker):
    cmd = "docker run --rm -v $(pwd):/data -w /data --entrypoint /usr/local/bin/subtree -it yijun/fast " + src_path + " " + tgt_path + " statements.csv"
    # cmd = "docker run --rm -v $(pwd):/e -it yijun/fast -p " + src_path + " " + tgt_path
    print(cmd)
    os.system(cmd)

def main():
    worker = args.worker
    input_path = args.input_path
    output_path = args.output_path
    
    Path(output_path).mkdir(parents=True, exist_ok=True)


    for subdir, dirs, files in os.walk(input_path):
        for project in dirs:
            raw_dir_path = os.path.join(subdir, project)
            target_path = os.path.join(output_path, project)
            Path(target_path).mkdir(parents=True, exist_ok=True)
            
            generate_folder_pb(raw_dir_path, target_path, worker)
         

                
            # print(fbs_path)

if __name__ == "__main__":
    main()              
     

