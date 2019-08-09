# This Python file uses the following encoding: utf-8

from w2v import Word2VecModel
import re
import multiprocessing
from tqdm import tqdm

def load_train_yuliao():
    tt = []
    with open("./datagrand/corpus.txt", mode="r", encoding="utf-8") as f1:
        for line in tqdm(f1.readlines()):
            # j = re.sub(r"\n", "", line)
            # j = re.sub(r"\r", "", j)
            # j = re.sub(r"\r\n", "", j)
            # j = re.sub(r"\n\r", "", j)
            j = line.strip()
            j = [st for st in j.split("_") if st != "" and st != "　" and st != " "]
            tt.append(j)
    return tt


if __name__ == "__main__":
    print("hello")
    vector_dim=200
    courpus = load_train_yuliao()
    #doc2vec_train(tt)
    w2vModel = Word2VecModel(size=vector_dim, window=8, min_count=10, workers=(multiprocessing.cpu_count()-2))
    w2vModel.train_model(courpus)
    w2vModel.save_model()
    #test()