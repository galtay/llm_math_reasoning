import json

fnames = [
    "output/meta-llama/Llama-3.1-8B-Instruct/output.json",
    "output/meta-llama/Llama-3.2-3B-Instruct/output.json",
]


for fname in fnames:
    with open(fname, "r") as fp:
        outputs = json.load(fp)
    print(fname)
    corrects = [el["correct"] for el in outputs]
    acc = sum(corrects) / len(corrects)
    print("num samples: ", len(corrects))
    print("accuracy: ", acc)
    print()
