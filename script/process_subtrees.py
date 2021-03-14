subtrees_path = "../subtrees/subtrees_raw.txt"
node_types_vocab_path = "../grammars/node_types_c_java_cpp_c-sharp.csv"

def load_node_types_vocab(path):
    node_types_dict = {}

    node_types = open(path).read().splitlines()
    for i, n in enumerate(node_types):
        node_types_dict[n] = i

    return node_types_dict

node_types_dict = load_node_types_vocab(node_types_vocab_path)

print(node_types_dict)

with open(subtrees_path, "r") as f:
    data = f.readlines()
    for line in data:
        line = line.replace("\n", "")
        line_splits = line.split(",")

        nodes = line_splits[:-1]
        depth = line_splits[-1]

        for node in nodes:
          
            try:
                node_splits = node.split("-")
                node_id = node_splits[0]
                node_type = node_splits[1]

                if node_type in node_types_dict:
                    node_type_index = node_types_dict[node_type]

                    print(node_type_index)
            except Exception as e:
                print(e)
