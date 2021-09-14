"""
    Name: db_helpers.py
    
    Description: Scripts to load the dataloaders for the result analysis
"""

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from speech.data_loader import init_wav2vec_feat, zero_pad_feats, check_audio_and_feat_shape, no_cut_or_pad_features, crop_utt_segments

import os, pickle
import numpy as np

import torch
from torch.utils import data

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print("Device: ", device)

def get_id_gt(utt_name):
    """ Get the speaker identity ground truth """
    gender = -1
    session = -1

    utt_splits = utt_name.split("/")
    for utt_split in utt_splits:
        if("Session" in utt_split):
            session = int(utt_split[(utt_split.find("Session") + len("Session")):])
        if(utt_split == "Female"):
            gender = 0
        elif(utt_split == "Male"):
            gender = 1

    if((gender < 0) or (session < 0)):
        print("Error! Session and gender were not correctly parsed")
        sys.exit(1)

    f_id = [0,2,4,6,8]
    m_id = [1,3,5,7,9]

    if gender == 0:
        return f_id[session-1]
    else:
        return m_id[session-1]

def get_wav2vec_index(config):
    """ Determine the index of wav2vec features """
    wav2vec_index = None
   
    if config.speech_input == "wav2vec":
        wav2vec_index = -4 # wav2vec is the last feature
    return wav2vec_index

def get_dataset_sample(audio, element, config, wav2vec_index, txt_emb):
    """ Return a dataset sample """
    new_wav2vec_index = None

    # x_real, emb_org, content_emb, emotion_gt
    if config.speech_input == "wav2vec":
        # use wav2vec features
        data_sample = [element[0], get_id_gt(utt_name=element[0]), audio, element[-2], element[2], element[3], element[-3], txt_emb, element[wav2vec_index]]
        new_wav2vec_index = -1
    else:
        # do not use wav2vec features
        data_sample = [element[0], get_id_gt(utt_name=element[0]), audio, element[-2], element[2], element[3], element[-3], txt_emb]

    return data_sample, new_wav2vec_index

def select_data_sample_to_return(db_config, features):
    """ Determines how to return the data sample """

    if db_config["speech_input"]=="wav2vec":
        return features["f_name"], features["id_gt"], features["uttr"], features["emb_org"], features["content_emb"], features["emotion_gt"], features["txt_emb"], features["wav2vec_feat"]
    elif db_config["speech_input"]=="spec":
        return features["f_name"], features["id_gt"], features["uttr"], features["emb_org"], features["content_emb"], features["emotion_gt"], features["txt_emb"]

def get_txt_emb(txt_feat_dir, f_name):
    """ Return huggingface embedding """
    max_seq_len = 122
    npy_file = f_name[:-4] + ".npy"

    try:
        txt_emb = np.load(txt_feat_dir + "/" + npy_file)
        txt_emb = txt_emb[0]
        txt_emb = txt_emb[1:-1] # ignore [CLS] and [SEP]
        # zero-pad the embedding
        txt_emb = np.pad(txt_emb, ((0,max_seq_len-txt_emb.shape[0]),(0,0)), mode="constant")
        txt_emb = torch.from_numpy(txt_emb)
    except:
        print("[WARNING] Could not read ", txt_feat_dir + "/" + npy_file)
        txt_emb = torch.from_numpy(np.empty([1], dtype=int))

    return txt_emb

class EvalUtterances(data.Dataset):
    """ Dataset class to make data loaders for the evaluation.
        This class assumes that the training and prediction occur
        over utterance segments but that the evaluation results
        should be acquired in the utterance level.
    """

    def __init__(self, data_dir, pkl_file, config, spk_dep=False):
        """Initialize and preprocess the Utterances dataset."""
        
        self.data_dir = data_dir
        self.len_crop = config.len_crop
        self.wav2vec_path = config.wav2vec_path
        self.speech_input = config.speech_input

        # get the metadata
        metaname = os.path.join(self.data_dir, pkl_file+".pkl")
        metadata = pickle.load(open(metaname, "rb"))

        # initialize the dataset
        dataset = [None] * len(metadata)

        self.wav2vec_index = get_wav2vec_index(config)

        # for all samples in the metadata
        for k, element in enumerate(metadata,0):
            # load the spectrogram
            try:
                audio = np.load(os.path.join(self.data_dir, element[0]))
            except:
                if spk_dep==False:
                    print("[ERROR] Unable to open the speech file ", os.path.join(self.data_dir, element[0]))
                    sys.exit(1)

            txt_emb = get_txt_emb(txt_feat_dir=config.txt_feat_dir,
                                  f_name=element[0])

            # check if audio and features have the same shape
            check_audio_and_feat_shape(audio=audio,
                                       element=element)
            
            # define the dataset sample
            dataset[k], new_wav2vec_index = get_dataset_sample(audio=audio,
                                                               element=element,
                                                               config=config,
                                                               wav2vec_index=self.wav2vec_index,
                                                               txt_emb=txt_emb)

        self.dataset = list(dataset)
        self.num_tokens = len(self.dataset)
        self.wav2vec_index = new_wav2vec_index
        
        print("num utterances: ", self.num_tokens) # number of utterances
        print("Finished loading the dataset...")
        
    def __getitem__(self, idx):
        """ Return a data sample (all the utterance's segments in a list). """
        dataset = self.dataset

        db_config = {}
        db_config["wav2vec_index"] = self.wav2vec_index
        db_config["len_crop"] = self.len_crop
        db_config["wav2vec_path"] = self.wav2vec_path
        db_config["speech_input"] = self.speech_input

        # pick a random utterance
        data_sample = dataset[idx]

        # get datasample basic features
        f_name, id_gt, spec, emb_org, phones, emotion_gt, _, txt_emb = data_sample[0:8]

        # initialize wav2vec features
        all_wav2vec_feat, wav2vec_feat = init_wav2vec_feat(db_config, f_name)

        features = {}
        features["f_name"] = f_name
        features["id_gt"] = id_gt
        features["spec"] = spec
        features["phones"] = phones
        features["emb_org"] = emb_org
        features["all_wav2vec_feat"] = all_wav2vec_feat
        features["emotion_gt"] = emotion_gt
        features["txt_emb"] = txt_emb

        # if the utterance is too short (zero-pad)
        if spec.shape[0] < self.len_crop:
            uttr, content_emb, wav2vec_feat = zero_pad_feats(features=features,
                                                             db_config=db_config,
                                                             as_list=True)
        # if the utterance is too long
        elif spec.shape[0] > self.len_crop:
            # get the number of segments
            uttr, content_emb, wav2vec_feat = crop_utt_segments(features, db_config)
        # the utterance has the exact crop size
        else:
            uttr, content_emb, wav2vec_feat = no_cut_or_pad_features(features, as_list=True)

        features["uttr"] = uttr
        features["content_emb"] = content_emb
        features["wav2vec_feat"] = wav2vec_feat

        return(select_data_sample_to_return(db_config=db_config,
                                            features=features))
    
    def __len__(self):
        """Return the number of utterances."""
        return self.num_tokens

############ Get data ############
def get_dbs(config, train_dir, test_dir, train_dir_cl, test_dir_cl):
    """ Return train and test data loaders """

    print("[INFO] Loading speaker independent data")
    train_db_utt = EvalUtterances(data_dir=train_dir,
                                  pkl_file="train",
                                  config=config)

    test_db_utt = EvalUtterances(data_dir=test_dir,
                                 pkl_file="test",
                                 config=config)

    print("[INFO] Loading speaker dependent data")
    train_db_cl_utt = EvalUtterances(data_dir=train_dir_cl,
                                      pkl_file="train",
                                      config=config,
                                      spk_dep=True)

    test_db_cl_utt = EvalUtterances(data_dir=test_dir_cl,
                                      pkl_file="test",
                                      config=config,
                                      spk_dep=True)

    # speaker-independent partitions (original data used to train the speech disentanglement model)
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    train_db = data.DataLoader(dataset=train_db_utt,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=0,
                                        drop_last=False,
                                        worker_init_fn=worker_init_fn)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    test_db = data.DataLoader(dataset=test_db_utt,
                              batch_size=1,
                              shuffle=False,
                              num_workers=0,
                              drop_last=False,
                              worker_init_fn=worker_init_fn)

    # speaker-dependent partitions (to train the classifiers)
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    train_db_cl = data.DataLoader(dataset=train_db_cl_utt,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=0,
                                        drop_last=False,
                                        worker_init_fn=worker_init_fn)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    test_db_cl = data.DataLoader(dataset=test_db_cl_utt,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    drop_last=False,
                                    worker_init_fn=worker_init_fn)

    return train_db, test_db, train_db_cl, test_db_cl