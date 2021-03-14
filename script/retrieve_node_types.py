import os
import json

home = "../tree-sitter-all-grammars"


for language_dir in os.listdir(home):
    # print(language_dir)
    language_dir = os.path.join(home, language_dir)
    for subdir , dirs, files in os.walk(language_dir): 
        for file in files:
            
            if file == "node-types.json":
                file_path = os.path.join(subdir, file)
                file_path_splits = file_path.split("/")
                lang = file_path_splits[2]
                lang = lang.split("-")[2:]
                lang = "-".join(lang)
                
                new_file_path = "node_types_{}.csv".format(lang)

                input_file = open(file_path)
                json_array = json.load(input_file)
                for item in json_array:
                    print(item["type"])
                    if item["type"] and len(item["type"])>0:
                        with open(new_file_path, "a") as f:
                            f.write(item["type"])
                            f.write("\n")
                    

