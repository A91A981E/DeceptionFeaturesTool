from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import os
import numpy as np
import pickle


def test():
    with open(os.path.join("output", "BERT_feature.pkl"), "rb") as f:
        dataset = pickle.load(f)
    with open(os.path.join("output", "BERT_feature_files.txt"), "r") as f:
        files = f.readlines()
    files = [i.strip() for i in files]
    labels = []
    for file in files:
        if "lie" in file:
            labels.append(1)
        else:
            labels.append(0)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dataset = np.array(dataset)
    labels = np.array(labels)
    acc = 0

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset, labels)):
        model = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(32, 32))
        model.fit(dataset[train_idx], labels[train_idx])
        pred = model.predict(dataset[test_idx])
        acc += accuracy_score(labels[test_idx], pred)
        print(accuracy_score(labels[test_idx], pred))
    print("#", acc / 5)


if __name__ == "__main__":
    test()
