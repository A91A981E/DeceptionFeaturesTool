import liwc
import re
from collections import Counter
import numpy as np
import os
from tqdm import tqdm
import pickle


def tokenize(text):
    for match in re.finditer(r"\w+", text, re.UNICODE):
        yield match.group(0)


class LIWC_feature:
    def __init__(self) -> None:
        self.parse, self.category_names = liwc.load_token_parser(
            os.path.join("feature_utils", "resources", "LIWC2015 Dictionary.dic")
        )
        self.category_names = sorted(self.category_names)

    def get_single_text_liwc_feature(self, text: str) -> np.array:
        feature = np.zeros(len(self.category_names))
        text = text.lower()
        text_tokens = tokenize(text)
        text_counts = Counter(
            category for token in text_tokens for category in self.parse(token)
        )
        for key in dict(text_counts):
            feature[self.category_names.index(key)] = text_counts[key]
        return feature

    def get_liwc_features_from_dir(self, dir_path: str) -> np.array:
        assert os.path.exists(dir_path), f"ERROR: '{dir_path}' does NOT exist."
        text_files = sorted([os.path.join(dir_path, i) for i in os.listdir(dir_path)])
        features = []
        for text_file in tqdm(text_files):
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            feature = self.get_single_text_liwc_feature(text)
            features.append(feature)
        features = np.stack(features)
        return features, text_files

    def get_liwc_features_from_dataset(self, dataset_path: str, output_path: str):
        assert os.path.exists(dataset_path), f"ERROR: '{dataset_path}' does NOT exist."
        try:
            dirs = sorted(
                [
                    os.path.join(dataset_path, "Transcription", i)
                    for i in os.listdir(os.path.join(dataset_path, "Transcription"))
                ]
            )
            features_all = []
            files_all = []
            for dir in dirs:
                features, files = self.get_liwc_features_from_dir(dir)
                features_all.append(features)
                files_all += files
            features_all = np.concatenate(features_all, axis=0)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(os.path.join(output_path, "LIWC_feature.pkl"), "wb") as f:
                pickle.dump(features_all, f)
            with open(os.path.join(output_path, "LIWC_feature_files.txt"), "w") as f:
                f.writelines("\n".join(files_all))
            return features_all
        except Exception as e:
            raise RuntimeWarning(
                f"Something happened when processing LIWC feature.\n{e}"
            )
        finally:
            print("Finish processing LIWC feature.")


if __name__ == "__main__":
    module = LIWC_feature()
    feature = module.get_liwc_features_from_dataset(
        os.path.join(
            "..",
            "RealLifeDeceptionDetection.2016",
            "Real-life_Deception_Detection_2016",
        ),
        "output",
    )
    print(type(feature))
