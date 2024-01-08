import os
from tqdm import tqdm


class OpenFace_feature:
    @staticmethod
    def get_openface_feature_from_single_video(file_path: str, output_path: str):
        assert os.path.exists(file_path), f"ERROR: '{file_path}' does NOT exist."
        extractor = os.path.join(
            "feature_utils",
            "algorithm",
            "OpenFace_2.2.0_win_x64",
            "FeatureExtraction.exe",
        )
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cmd = rf'{extractor} -f "{file_path}" -out_dir "{output_path}" -pose -gaze -aus -2Dfp -3Dfp'
        print(cmd)
        os.system(cmd)

    @staticmethod
    def get_openface_feature_from_dir(dir_path: str, output_path: str):
        assert os.path.exists(dir_path), f"ERROR: '{dir_path}' does NOT exist."
        files = sorted([os.path.join(dir_path, i) for i in os.listdir(dir_path)])
        for file in tqdm(files):
            OpenFace_feature.get_openface_feature_from_single_video(
                file,
                os.path.join(
                    output_path, "Deceptive" if "Deceptive" in dir_path else "Truthful"
                ),
            )

    @staticmethod
    def get_openface_feature_from_dataset(dataset_path: str, output_path: str):
        assert os.path.exists(dataset_path), f"ERROR: '{dataset_path}' does NOT exist."
        try:
            dirs = sorted(
                [
                    os.path.join(dataset_path, "Clips", i)
                    for i in os.listdir(os.path.join(dataset_path, "Clips"))
                ]
            )
            for dir in dirs:
                OpenFace_feature.get_openface_feature_from_dir(
                    dir, os.path.join(output_path, "openface")
                )
        except Exception as e:
            raise RuntimeWarning(
                f"Something wrong happened when processing openface features.\n{e}"
            )
        finally:
            print("Finish processing OpenFace features.")


if __name__ == "__main__":
    OpenFace_feature.get_openface_feature_from_dataset(
        r"../RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016",
        "output",
    )
