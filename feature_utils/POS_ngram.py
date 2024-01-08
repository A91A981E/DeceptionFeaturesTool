import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import numpy as np
import os
import pickle
from tqdm import tqdm

nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("stopwords")

POS_TAG = [
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NN",
    "NNS",
    "NNP",
    "NNPS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
]


class POS_Feature:
    def __init__(self):
        self.stopword = set(stopwords.words("english") + list(punctuation))

    def get_sentence_pos_feature(self, s):
        feature = np.zeros(len(POS_TAG))
        words_before = nltk.word_tokenize(s.lower())
        words = []
        for w in words_before:
            if w not in self.stopword:
                words.append(w)
        pos_tags = nltk.pos_tag(words)
        for word, pos in pos_tags:
            if pos in POS_TAG:
                feature[POS_TAG.index(pos)] += 1
        return feature

    def get_pos_feature_from_dir(self, dir_path):
        assert os.path.exists(dir_path), f"ERROR: '{dir_path}' does NOT exist."
        files = sorted([os.path.join(dir_path, i) for i in os.listdir(dir_path)])
        features = []
        for file in tqdm(files):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read().strip().lower()
            feature = self.get_sentence_pos_feature(text)
            features.append(feature)
        features = np.stack(features)
        return features, files

    def get_pos_feature_from_dataset(self, dataset_path: str, output_path: str):
        assert os.path.exists(dataset_path), f"ERROR: '{dataset_path}' does NOT exist."
        try:
            dirs = sorted(
                [
                    os.path.join(dataset_path, "Transcription", i)
                    for i in os.listdir(os.path.join(dataset_path, "Transcription"))
                ]
            )
            pos_features = []
            files_all = []
            for dir in dirs:
                features, files = self.get_pos_feature_from_dir(dir)
                pos_features.append(features)
                files_all += files
            pos_features = np.concatenate(pos_features, axis=0)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(os.path.join(output_path, "pos_feature.pkl"), "wb") as f:
                pickle.dump(pos_features, f)
            with open(os.path.join(output_path, "pos_feature_files.txt"), "w") as f:
                f.writelines("\n".join(files_all))
            return pos_features
        except Exception as e:
            raise RuntimeWarning(
                f"Something happened when processing POS feature.\n{e}"
            )
        finally:
            print("Finish processing POS feature.")


if __name__ == "__main__":
    m = POS_Feature()
    a = m.get_pos_feature_from_dataset(
        os.path.join(
            "..",
            "RealLifeDeceptionDetection.2016",
            "Real-life_Deception_Detection_2016",
        ),
        "output",
    )
    print(a)
