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
                node_types = []
                for item in json_array:
                    # print(item["type"])
                    if item["type"] and len(item["type"])>0 and item["type"] != "\n":
                        node_types.append(item["type"])
                 

                node_types = list(set(node_types))
                node_types = list(filter(None, node_types))

                with open(new_file_path, "w") as f:
                    for n in node_types:
                        f.write(n)
                        f.write("\n")
                    

