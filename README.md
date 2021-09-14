# Multimodal Emotion Recognition

This repository contains the code used to perform the training and the analysis of the models described in the paper "Multimodal Emotion Recognition with High-level Speech and Text Features" accepted in the ASRU 2021 conference.

## Repository organization

This repository is organized as follows:

- aligner: contains a script used to extract phone alignment information using [Gentle](https://github.com/lowerquality/gentle). This information is further used by the scripts in the `speech` folder.

- analysis: contains the scripts used to obtain the fused modalities (speech and text) results, and to get detailed results for each modality.

- speech: contains the scripts to process the speech files, and to train and evaluate the speech emotion recognition model.

- text: contains the scripts to extract text features using pre-trained Transformer-based models, and to train a text emotion recognition model over these features.

- wav2vec: contains the scripts to extract wav2vec features from the speech files (these features are further used by the scripts in the `speech` folder).

## Dataset

We evaluated our method with the IEMOCAPdataset. Therefore, the scripts were written assuming it is possible to organize the data in 5 sessions, and according to the speaker's gender. To request access to the IEMOCAP dataset, and, to know more about this dataset, refer to the [IEMOCAP webpage](https://sail.usc.edu/iemocap/) .

## How to run the scripts

For information on how to run the scripts step-by-step, please refer to `how_to.md`, and, for information on the requirements to run the scripts of this repository, please refer to `requirements.md`.

## More information

The speech model found in the folder `speech` was inspired by the AutoVC model. For more information on AutoVC, check their [github repository](https://github.com/auspicious3000/autovc) and [paper](https://arxiv.org/abs/1905.05879).

For more information on the technical components of our method, on the results, and on the discussion of these results, please refer to our paper.
