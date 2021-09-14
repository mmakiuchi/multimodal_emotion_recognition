# Required data and label files

- In the instructions written in this file, we assume that the emotion labels are in `/data/labels`, and that the speech and text data are in `/data/speech` and `/data/text`, respectively.

- We also assume that you are using the IEMOCAP dataset.

## Emotion label files (`/data/labels`)

 - We consider only the utterances that are given the same label by at least two annotators.

 - The original "Excited" and "Happy" classes should be merged into the "Happy" class beforehand.
 
 - You should keep only the label files corrsponding to the 4 emotion classes (Angry, Happy+Excited, Neutral and Sad).

## Speech and text data (`/data/speech` and `/data/text`)

 - The speech files should be organized by session and gender (e.g. `/data/speech/Session1/Female`, `/data/speech/Session1/Male`, `/data/speech/Session2/Female`, etc)

 - The text files should also be organized according to session and gender

 - You should keep only the data corrsponding to the 4 emotion classes (Angry, Happy+Excited, Neutral and Sad).

In total, there should be data (speech and text) and labels for 5,531 utterances.

# Train speech model
`cd speech`

### Create the spectrograms and save them in spmel_dir

`python make_spect.py --spmel_dir "spmels" --speech_dir "/data/speech"`

By running this command, mel-frequency spectrograms are extracted from the speech files in `speech_dir`, and they are stored in the directory specified by `spmel_dir`. These spectrograms are later used during the speech emotion recognition model training.

### Organize the spectrograms in folds
In this part, the extracted spectrograms are organized in 5 speaker independent folds for a leave-one-session-out cross-validation, by running:

`python organize_folds.py --spmel_dir "spmels" --folds_dir "folds"`

### Extract the Gentle aligner json files and store them in `aligner/align_results`.
For this part, you should have the Gentle files and folders (`exp`, `ext`, `gentle`, `align.py`, etc) in the `aligner` folder (check the Gentle repository for more information on how to use the Gentle aligner).

After having the Gentle aligner set up, you can extract the alignment information for all the text files by running:

`cd ../aligner`

`python extract_phonemes.py --speaker "M1i" --txt_dir "/data/text" --audio_dir "/data/speech"`

`python extract_phonemes.py --speaker "M1s" --txt_dir "/data/text" --audio_dir "/data/speech"`

`python extract_phonemes.py --speaker "F1i" --txt_dir "/data/text" --audio_dir "/data/speech"`

`python extract_phonemes.py --speaker "F1s" --txt_dir "/data/text" --audio_dir "/data/speech"`


`python extract_phonemes.py --speaker "M2i" --txt_dir "/data/text" --audio_dir "/data/speech"`

`python extract_phonemes.py --speaker "M2s" --txt_dir "/data/text" --audio_dir "/data/speech"`

`python extract_phonemes.py --speaker "F2i" --txt_dir "/data/text" --audio_dir "/data/speech"`

`python extract_phonemes.py --speaker "F2s" --txt_dir "/data/text" --audio_dir "/data/speech"`

...

`python extract_phonemes.py --speaker "M5i" --txt_dir "/data/text" --audio_dir "/data/speech"`

`python extract_phonemes.py --speaker "M5s" --txt_dir "/data/text" --audio_dir "/data/speech"`

`python extract_phonemes.py --speaker "F5i" --txt_dir "/data/text" --audio_dir "/data/speech"`

`python extract_phonemes.py --speaker "F5s" --txt_dir "/data/text" --audio_dir "/data/speech"`

`cd ..`

### Create the pkl files for all folds and for both train and test partitions

For this part, it is necessary to have a "dictionary of phones" (i.e., a phone and its corresponding id)
The dictionary that I used is `aligner/phone_dict.csv`, but you should check if the phones are consistent
with your alignment results, otherwise there will be errors when running the code.

After you get the phone dictionary, you can run the following to generate the pkl files:

`python speech/make_pkl_data.py --mode "train" --spmel_dir "speech/folds/fold1/train" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "train" --spmel_dir "speech/folds/fold2/train" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "train" --spmel_dir "speech/folds/fold3/train" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "train" --spmel_dir "speech/folds/fold4/train" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "train" --spmel_dir "speech/folds/fold5/train" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "test" --spmel_dir "speech/folds/fold1/test" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "test" --spmel_dir "speech/folds/fold2/test" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "test" --spmel_dir "speech/folds/fold3/test" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "test" --spmel_dir "speech/folds/fold4/test" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "test" --spmel_dir "speech/folds/fold5/test" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`


### Create the output directories

`cd speech`

`mkdir results`

`cd results`

`mkdir fold1`

`mkdir fold2`

`mkdir fold3`

`mkdir fold4`

`mkdir fold5`

`cd ../../`

These directories (`speech/results/fold1`, etc) will store speech model checkpoints and training logs for each training fold.

### Extract the wav2vec features and save them in `wav2vec/features/large-lv60/all_hidden_states`

To run this step, you will need to have a csv file in `wav2vec/utils` with two columns:
- The path to a speech utterance file
- The utterance text

The columns should be separated by a semicolon (`;`) and the csv should have a row for each sample in the dataset. Remember to transform the utterance text to all caps and to remove the special tokens `[LAUGHER]`, `[BREATHING]`, etc from the text in the csv file. For more information, check Huggingface datasets.

After you make this csv file, you will be able to run:

`python wav2vec/extract_wav2vec.py`

which will extract the wav2vec features.

### Running speech training script for few iterations

With the following command, you can run the speech model training for few iterations (specified by `num_iters`) to check if everything is running fine:

`python speech/main.py --reproduct_mode "free" --num_iters 500 --checkpoint_step 250`

If you want to check or change the model's parameters, please refer to the scripts `speech/parser_helper.py` and `speech/speech_model.py`. The evaluation and training loops are in `speech/solver.py`

### Reproducing the experiments with the small bottleneck (main experiments)

You can also reproduce the experiments found in our paper for all the 5 folds, by running the following commands (be aware that the training for each fold may take a long time depending on the GPU that you are using):

`python speech/main.py --reproduct_mode "small" --train_dir "speech/folds/fold1/train" --test_dir "speech/folds/fold1/test" --out_dir "speech/results/fold1" --wav2vec_path "wav2vec/features/large-lv60/all_hidden_states"`

`python speech/main.py --reproduct_mode "small" --train_dir "speech/folds/fold2/train" --test_dir "speech/folds/fold2/test" --out_dir "speech/results/fold2" --wav2vec_path "wav2vec/features/large-lv60/all_hidden_states"`

`python speech/main.py --reproduct_mode "small" --train_dir "speech/folds/fold3/train" --test_dir "speech/folds/fold3/test" --out_dir "speech/results/fold3" --wav2vec_path "wav2vec/features/large-lv60/all_hidden_states"`

`python speech/main.py --reproduct_mode "small" --train_dir "speech/folds/fold4/train" --test_dir "speech/folds/fold4/test" --out_dir "speech/results/fold4" --wav2vec_path "wav2vec/features/large-lv60/all_hidden_states"`

`python speech/main.py --reproduct_mode "small" --train_dir "speech/folds/fold5/train" --test_dir "speech/folds/fold5/test" --out_dir "speech/results/fold5" --wav2vec_path "wav2vec/features/large-lv60/all_hidden_states"`

# Training the text model

`cd text`

`mkdir txt_model_out`

`cd ..`

### Extract text features and save them in text/last_hidden_state/bert-large-uncased

`python text/feature_extractor.py --text_dir "/data/text" --label_dir "/data/labels"`

### Train the text model

`python text/train_txt_model.py --label_dir "/data/labels"`

# Evaluate results
You need to organize the spectrograms in speaker dependent folds to train the speaker identity classifiers.
Then, you should generate pkl files for the speaker dependent folds, so that the data can be read.
We will assume that the spk dependent folds are in `speech/spk_dep_folds`.

The analysis script outputs accuracy results and LDA plots for all the 5 folds.
The speech and text models used for each fold are defined in the main function of analysis/analysis_main.py,
and other model parameters are defined in analysis/config_helpers.py.
You should change these two scripts according to your experiments and model checkpoints.

The input variables to `analysis/analysis_main.py` are:
- speech_dir : directory to the pkl files for each fold and partition (train and test)
- speech_dir_cl : directory to the speaker dependent partitions with the pkl files
- wav2vec_dir : directory where the wav2vec features are stored
- txt_feat_dir : directory to the BERT text features
- label_path : path to the dataset labels

With the speaker dependent folds data set up, you can run:

`python analysis/analysis_main.py --speech_dir "speech/folds" --speech_dir_cl "speech/spk_dep_folds" --wav2vec_dir "wav2vec/features/large-lv60/all_hidden_states" --txt_feat_dir "text/last_hidden_state/bert-large-uncased" --label_path "data/labels"`


# Extra: how to get pkl data from speaker dependent folds

`python speech/make_pkl_data.py --mode "train" --spmel_dir "speech/spk_dep_folds/fold1/train" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "train" --spmel_dir "speech/spk_dep_folds/fold2/train" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "train" --spmel_dir "speech/spk_dep_folds/fold3/train" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "train" --spmel_dir "speech/spk_dep_folds/fold4/train" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "train" --spmel_dir "speech/spk_dep_folds/fold5/train" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`


`python speech/make_pkl_data.py --mode "test" --spmel_dir "speech/spk_dep_folds/fold1/test" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "test" --spmel_dir "speech/spk_dep_folds/fold2/test" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "test" --spmel_dir "speech/spk_dep_folds/fold3/test" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "test" --spmel_dir "speech/spk_dep_folds/fold4/test" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`

`python speech/make_pkl_data.py --mode "test" --spmel_dir "speech/spk_dep_folds/fold5/test" --dict_file "aligner/phone_dict.csv" --labels_dir "/data/labels" --wav_dir "/data/speech" --txt_dir "/data/text" --phone_dir "aligner/align_results"`