
with open("../../subtrees.csv", "r") as f:
    data = f.read().splitlines()
    for l in data:
        if "ERROR" not in l:
            subtree = l.split("-")
            if len(subtree) > 3:
                with open("../../subtrees_larger_than_3.csv", "a") as f1:
                    f1.write(l)
                    f1.write("\n")