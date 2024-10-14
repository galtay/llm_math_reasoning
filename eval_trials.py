import json

fnames = [
    "output/meta-llama/Llama-3.1-8B-Instruct/output.json",
    "output/meta-llama/Llama-3.2-3B-Instruct/output.json",
]

outputs = []
for fname in fnames:
    with open(fname, "r") as fp:
        out1 = json.load(fp)
    print(fname)
    pieces = fname.split("/")
    model = pieces[-2]
    corrects = [el["correct"] for el in out1]
    acc = sum(corrects) / len(corrects)
    for el in out1:
        el["model"] = model
    print("num samples: ", len(corrects))
    print("accuracy: ", acc)
    print()
    outputs.extend(out1)

with open("combined_outputs.json", "w") as fp:
    json.dump(outputs, fp)