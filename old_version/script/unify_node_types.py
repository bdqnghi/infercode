import os

home = "../grammars"

langs = ["node_types_c.csv", "node_types_java.csv", "node_types_cpp.csv", "node_types_c-sharp.csv", "node_types_rust.csv"]
all_node_types = []
for subdir , dirs, files in os.walk(home): 
    for file in files:
        if file in langs:
            file_path = os.path.join(subdir, file)
            with open(file_path, "r") as f:
                data = f.readlines()
                for line in data:
                    all_node_types.append(line.replace("\n",""))

all_node_types = list(set(all_node_types))

for n in all_node_types:
    with open("../grammars/node_types_c_java_cpp_c-sharp_rust.csv", "a") as f1:
        if n and len(n)>0: 
            f1.write(n)
            f1.write("\n")