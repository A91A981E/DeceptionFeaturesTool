from .algorithm.acoustic_feature import (
    Spectrogram,
    SpectrumFeatures,
    VAD,
    RhythmFeatures,
    OpenSmileFeatureSet,
    QualityFeatures,
)
import os
from tqdm import tqdm
import pickle


class MFCC_feature:
    def __init__(self) -> None:
        pass

    def get_single_file_MFCC_feature(self, wav_file):
        assert os.path.exists(wav_file), f"ERROR: '{wav_file} does NOT exist."
        spectrum_features = SpectrumFeatures(wav_file)
        mfcc = spectrum_features.mfcc(n_mfcc=13, ceplifter=22, n_mels=26)
        return mfcc

    def get_MFCC_features_from_dir(self, dir_path):
        assert os.path.exists(dir_path), f"ERROR: '{dir_path} does NOT exist."
        files = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]
        files = sorted(files)
        mfcc_feature = []
        for file in tqdm(files):
            mfcc = self.get_single_file_MFCC_feature(file)
            mfcc_feature.append(mfcc)
        return mfcc_feature, files

    def get_MFCC_features_from_dataset(self, dataset_path, output_path):
        assert os.path.exists(dataset_path), f"ERROR: '{dataset_path} does NOT exist."
        try:
            dirs = sorted(
                [
                    os.path.join(dataset_path, "Audios", i)
                    for i in os.listdir(os.path.join(dataset_path, "Audios"))
                ]
            )
            features_all = []
            files_all = []
            for dir in dirs:
                features, files = self.get_MFCC_features_from_dir(dir)
                features_all += features
                files_all += files
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(os.path.join(output_path, "MFCC_output.pkl"), "wb") as f:
                pickle.dump(features_all, f)
            with open(os.path.join(output_path, "MFCC_output_files.txt"), "w") as f:
                f.writelines("\n".join(files_all))
            return features_all
        except Exception as e:
            raise RuntimeWarning(
                "Something happened when processing OpenSMILE features.\n{e}"
            )
        finally:
            print("Finish processing OpenSMILE features.")


if __name__ == "__main__":
    module = MFCC_feature()
    features = module.get_MFCC_features_from_dataset(
        os.path.join(
            "..",
            "RealLifeDeceptionDetection.2016",
            "Real-life_Deception_Detection_2016",
        ),
        "output",
    )
    print(type(features))
