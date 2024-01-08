from .liwc_utils import LIWC_feature
from .openface_utils import OpenFace_feature
from .mfcc_utils import MFCC_feature
from .POS_ngram import POS_Feature
from .bert_feature import BERT_feature
from .opensmile_utils import AudioProcessor

from typing import Optional, List, Union
import os


class MultiModalFeatureExtraction:
    def __init__(
        self, output_path: Optional[str], modalities: Union[List, set], **kwargs
    ) -> None:
        if isinstance(modalities, list):
            modalities = set(modalities)
        self.modalities = modalities
        if output_path is None:
            output_path = os.path.join("output")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = output_path

    def extract(self, dataset_path: str):
        assert os.path.exists(dataset_path), f"ERROR: '{dataset_path}' does NOT exist."

        # OpenFace feature
        if "openface" in self.modalities:
            OpenFace_feature.get_openface_feature_from_dataset(
                dataset_path, self.output_path
            )

        # MFCC feature
        if "mfcc" in self.modalities:
            MFCC_feature().get_MFCC_features_from_dataset(
                dataset_path, self.output_path
            )

        # LIWC feature
        if "liwc" in self.modalities:
            LIWC_feature().get_liwc_features_from_dataset(
                dataset_path, self.output_path
            )
        # POS feature
        if "pos" in self.modalities:
            POS_Feature().get_pos_feature_from_dataset(dataset_path, self.output_path)

        # BERT feature
        if "bert" in self.modalities:
            BERT_feature().get_bert_features_from_dataset(
                dataset_path, self.output_path
            )

        # OpenSMILE feature
        if "opensmile" in self.modalities:
            AudioProcessor(
                dataset_path, output_folder=os.path.join(self.output_path, "OpenSMILE")
            )


if __name__ == "__main__":
    feature_to_extract = ["openface", "mfcc", "liwc", "pos", "bert", "opensmile"]
    output_dir = "output"

    extractor = MultiModalFeatureExtraction(output_dir, feature_to_extract)
    extractor.extract(
        r"D:\Postgraduate\Motion\Deception Detection\RealLifeDeceptionDetection.2016\Real-life_Deception_Detection_2016"
    )
