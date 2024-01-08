# Deception Detection Multimodal Feature Extraction And Significance Test Tool

> version 2.0

+ [X] Gaze (OpenFace)
+ [X] Head Pose (OpenFace)
+ [X] Action Units (OpenFace)
+ [X] 68 Landmarks in 2D/3D (OpenFace)
+ [X] MFCC (OpenSMILE)
+ [X] LIWC (LIWC-2015)
+ [X] BERT Embedding (Hugging Face version 🤗)
+ [X] POS (NLTK)
+ [X] OpenSMILE

## Co-workers

Zihan Ji (South China University of Technology), Mengxi Gao (South China University of Technology), Lutao Yan (South China University of Technology)

## Preparation

1. Download OpenFace2.2.0 from [https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_2.2.0/OpenFace_2.2.0_win_x64.zip](https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_2.2.0/OpenFace_2.2.0_win_x64.zip).
2. Unzip the file at `feature_utils/algorithm`.
3. Open PowerShell at `feature_utils/algorithm/OpenFace_2.2.0_win_x64` and execute `download_models.ps1` (VPN needed for China mainland users).
4. Arrange dataset's format as Reallife-trial-dataset:
   ```bash
   D:.
   └─Real-life_Deception_Detection_2016
       ├─Clips
       │  ├─Deceptive
       │  └─Truthful
       ├─Audios
       │  ├─Deceptive
       │  └─Truthful
       └─Transcription
           ├─Deceptive
           └─Truthful
   ```
5. Install PyTorch from [https://pytorch.org](https://pytorch.org/) for BERT.
6. Install necessary requirements by `pip install -r requirements.txt` (Python 3.8).

## Usage

### Extract features all at once

This is an example code for using this tool.

```python
from feature_utils import MultiModalFeatureExtraction

extractor = MultiModalFeatureExtraction("output", ["openface", "mfcc", "liwc", "pos", "bert", "opensmile"])
extractor.extract(
    r"PATH/TO/DATASET"
)
```

The results would be saved to the path you set.

### Extract single feature

Or you may choose to extract one particular feature.

```python
from feature_utils import (
    LIWC_feature
    OpenFace_feature,
    MFCC_feature,
    POS_Feature,
    BERT_feature,
    AudioProcessor
)


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
    AudioProcessor(dataset_path)
```

And the result would be saved to `output_path`, respectively.

### Significance Test

In our new version, we conduct significance test using t-test. All features extracted above would be tested. By the way, as there have been amount of methods on using OpenFace csv file results, we try to analyze eye movement and Action Units, firstly. Final version is coming soon.
