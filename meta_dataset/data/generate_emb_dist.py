import numpy as np
import json
from sklearn import metrics
import csv
from tqdm import tqdm
import sys

words = np.loadtxt("/mnt/disks/0/imagenet/ILSVRC2012_img_train/words.txt", delimiter='\t', dtype=str)
words = {k:v for k, v in words}
print(len(words.keys()))

with open("/mnt/disks/0/records/mini_imagenet/dataset_spec.json") as f:
    class_names = json.load(f)['class_names']
print(class_names)

embeddings = {}
with open("/mnt/disks/0/imagenet/ILSVRC2012_img_train/numberbatch-en-19.08.txt") as f:
    reader = csv.reader(f, delimiter=' ')
    next(reader)
    for row in tqdm(reader):
        embeddings[row[0]] = np.array(row[1:])
print(len(embeddings.keys()))

num_classes = len(class_names.keys())
emb_dim = len(list(embeddings.values())[0])
class_embeddings = np.zeros((num_classes, emb_dim))
for i, class_name in class_names.items():
    print(i)
    names = [w.strip() for w in words[class_name].split(',')]
    print(names)
    if words[class_name] in embeddings.keys():
        class_embeddings[int(i)] = embeddings[words[class_name]]
    else:
        candidates = []
        for ename, e in embeddings.items():
            if ename in names:
                print(ename)
                candidates.append(e)
        if candidates:
            class_embeddings[int(i)] = np.mean(np.array(candidates, dtype=np.float64), axis=0)
        else:
            tokens = [n.replace(',', '').lower() for n in words[class_name].split(' ')]
            print(tokens)
            class_embeddings[int(i)] = np.mean(np.array([embeddings[t] for t in tokens if t in embeddings], dtype=np.float64), axis=0)

class_embeddings = np.array(class_embeddings)
np.set_printoptions(threshold=sys.maxsize)

sims = metrics.pairwise.cosine_similarity(class_embeddings)
print(sims.shape)

np.savetxt("/mnt/disks/0/records/mini_imagenet/sims.tsv", sims, delimiter='\t')
