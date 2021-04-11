import pickle
tree_path = "java-small-test/buckets-all-training.pkl"
data = pickle.load(open(tree_path, "rb"))
for k, v in data.items():
    print("Key", k)
    print("Len v", len(v))
    print(v)
