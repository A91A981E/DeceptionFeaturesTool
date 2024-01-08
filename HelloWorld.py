from feature_utils import MultiModalFeatureExtraction

# feature_to_extract = ["openface", "mfcc", "liwc", "pos", "bert", "opensmile"]
feature_to_extract = ["openface", "mfcc"]
output_dir = "DOLOS_output"

extractor = MultiModalFeatureExtraction(output_dir, feature_to_extract)
extractor.extract(
    "PATH/TO/DATASET"
)
