import json

fnames = [
    "llama3.2-3b-instruct-fp16-output.json",
    "llama3.1-8b-instruct-fp16-output.json",
]


for fname in fnames:
    with open(fname, "r") as fp:
        outputs = json.load(fp)
    print(fname)
    corrects = [el["correct"] for el in outputs]
    acc = sum(corrects) / len(corrects)
    print("accuracy: ", acc)
    print()
