from transformers import BertModel, BertTokenizer
import torch
import os
import numpy as np
from tqdm import tqdm
import pickle
import re


class BERT_feature:
    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(r"bert-base-uncased")
        self.model = BertModel.from_pretrained(r"bert-base-uncased")
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def get_bert_feature_from_text(self, text: str):
        text = "[SEP]".join(re.findall("(.*?\.)\ ?", text))
        tokens = self.tokenizer.encode_plus(
            text, add_special_tokens=True, return_tensors="pt"
        )

        for key, value in tokens.items():
            tokens[key] = value.to(self.device)
        output = self.model(**tokens)
        cls_embedding = output.last_hidden_state[0, 0, :]
        return cls_embedding.cpu().numpy()

    def get_bert_feature_from_dir(self, dir_path: str):
        assert os.path.exists(dir_path), f"ERROR: '{dir_path}' does NOT exist."
        text_files = sorted([os.path.join(dir_path, i) for i in os.listdir(dir_path)])
        features = []
        for text_file in tqdm(text_files):
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read().strip().lower()
            feature = self.get_bert_feature_from_text(text)
            features.append(feature)
        features = np.stack(features)
        return features, text_files

    def get_bert_features_from_dataset(self, dataset_path: str, output_path: str):
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
                features, files = self.get_bert_feature_from_dir(dir)
                features_all.append(features)
                files_all += files
            features_all = np.concatenate(features_all, axis=0)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(os.path.join(output_path, "BERT_feature.pkl"), "wb") as f:
                pickle.dump(features_all, f)
            with open(os.path.join(output_path, "BERT_feature_files.txt"), "w") as f:
                f.writelines("\n".join(files_all))
            return features_all
        except Exception as e:
            raise RuntimeWarning(
                f"Something happened when processing BERT feature.\n{e}"
            )
        finally:
            print("Finish processing BERT feature.")


if __name__ == "__main__":
    m = BERT_feature()
    a = m.get_bert_features_from_dataset(
        os.path.join(
            "..",
            "RealLifeDeceptionDetection.2016",
            "Real-life_Deception_Detection_2016",
        ),
        "output",
    )
    print(a.shape)
