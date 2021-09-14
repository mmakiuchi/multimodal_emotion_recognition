"""
    Name: config_helpers.py
    
    Description: Scripts to define the configuration variables
"""

import argparse

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_sp_config(config):
    sp_config = {}

    sp_config["wav2vec_path"] = config.wav2vec_dir

    # Model configuration
    sp_config["dim_emb"] = 256
    sp_config["dim_phone"] = 128
    sp_config["dim_phone_emb"] = 128
    sp_config["dim_pre"] = 512
    sp_config["use_drop"] = True
    sp_config["num_emo_classes"] = 4

    sp_config["dim_neck"] = 8
    sp_config["freq"] = 48

    sp_config["len_crop"] = 96
    sp_config["num_mels"] = 80
    sp_config["wav2vec_feat_len"] = 1024
    sp_config["speech_input"] = "wav2vec"
    
    # Huggingface text embeddings
    sp_config["txt_feat_dir"] = config.txt_feat_dir

    sp_config["label_path"] = config.label_path

    return dotdict(sp_config)

def get_data_dirs(config):

    data_dirs = {}

    # speaker idependent folds (same ones used to train the speech and text models)
    data_dirs["data_dir"] = config.speech_dir

    # speaker dependent folds (to train the classifiers for speaker recognition)
    data_dirs["data_dir_cl"] = config.speech_dir_cl

    return data_dirs
    
def argparser():
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument("--speech_dir", type=str, help="Speech data for speaker independent folds")
    parser.add_argument("--speech_dir_cl", type=str, help="Speech data for speaker dependent folds")
    parser.add_argument("--wav2vec_dir", type=str, help="Directory for wav2vec features")
    parser.add_argument("--txt_feat_dir", type=str, help="Directory to text features")
    parser.add_argument("--label_path", type=str, help="Path to the label files")

    config = parser.parse_args()

    return config

def get_config():

    config = argparser()
    sp_config = get_sp_config(config)
    data_dirs = get_data_dirs(config)

    return sp_config, data_dirs