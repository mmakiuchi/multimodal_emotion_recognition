"""
    Name: data_loader.py
    Description: Loads the data for training and testing.
"""

from torch.utils import data
import torch
import numpy as np
import pickle 
import os, math, sys

def get_word_seq_cut(word_seq, begin, end, word_intervals):
    """ Return the word sequence cut from begin to end. """
    word_seq_cut = []
    if not (word_seq == [None]):
        for word, interval in zip(word_seq, word_intervals):
            word_begin = interval[0]
            word_end = interval[1]
            if((word_begin >= begin) and (word_end <= end)):
                word_seq_cut.append(word)
    return word_seq_cut

def get_wav2vec_index(config):
    """ Determine the index of wav2vec features """
    wav2vec_index = None
   
    if config.speech_input=="wav2vec":
        wav2vec_index = -4 # wav2vec is the last feature
    return wav2vec_index

def check_audio_and_feat_shape(audio, element):
    """ Check if the audio shape matches the features shape """
    if not (audio.shape[0] == element[2].shape[0]):
        print("[ERROR] Audio and main phone sequence do not have the same size")
        sys.exit(1)

def get_dataset_sample(audio, element, config):

    """ Return a dataset sample """
    new_wav2vec_index = None
    wav2vec_feat = None # we will load the features later

    if config.speech_input=="wav2vec":
        # use wav2vec features
        data_sample = [element[0], audio, element[-2], element[2], element[3], element[-3], wav2vec_feat]
        new_wav2vec_index = -1
    elif config.speech_input=="spec":
        # do not use wav2vec features
        data_sample = [element[0], audio, element[-2], element[2], element[3], element[-3]]

    return data_sample, new_wav2vec_index

def get_wav2vec_tensor(file_name, wav2vec_path):
    """ Get wav2vec features from tensor """
    wav2vec_feat = np.load(os.path.join(wav2vec_path, file_name[:-4] + ".npy"))
    return wav2vec_feat

def init_wav2vec_feat(db_config, file_name):
    """ Initialize wav2vec features """
    wav2vec_feat = None
    all_wav2vec_feat = None

    if db_config["speech_input"]=="wav2vec":
        all_wav2vec_feat = get_wav2vec_tensor(file_name=file_name,
                                              wav2vec_path=db_config["wav2vec_path"])

    return all_wav2vec_feat, wav2vec_feat

def zero_pad(array, len_pad):
    """ Zero pads a 2d array with zeros to the right """
    return(np.pad(array, ((0,len_pad),(0,0)), "constant"))

def zero_pad_feats(features, db_config, as_list):
    """ Zero pads the features that are too short"""

    wav2vec_feat = None
    len_pad = db_config["len_crop"] - features["spec"].shape[0]
    uttr = zero_pad(features["spec"], len_pad)
    content_emb = zero_pad(features["phones"], len_pad)
    
    if db_config["speech_input"]=="wav2vec":
        wav2vec_feat = np.pad(features["all_wav2vec_feat"], ((0,0),(0,len_pad),(0,0)), "constant")

    if as_list:
        return [uttr], [content_emb], [wav2vec_feat]
    else:
        return uttr, content_emb, wav2vec_feat

def feats_crop(features, left, db_config):
    """ Crops the features from a starting point to the left """
    wav2vec_feat = None
    uttr = features["spec"][left:left+db_config["len_crop"], :]
    content_emb = features["phones"][left:left+db_config["len_crop"], :]
    if db_config["speech_input"]=="wav2vec":
        wav2vec_feat = features["all_wav2vec_feat"][:, left:left+db_config["len_crop"], :]
    
    return uttr, content_emb, wav2vec_feat

def random_feats_crop(features, db_config):
    """ Crops the features randomly """
    # randomly crop the utterance
    left = np.random.randint(features["spec"].shape[0]-db_config["len_crop"])
    return(feats_crop(features, left, db_config))
    
def begin_feats_crop(features, db_config):
    """ Crops the fetaures from the beginning """
    return(feats_crop(features, 0, db_config))

def crop_utt_segments(features, db_config):
    """ For all the features, crop all the segments in an utterance
        and return them as lists """
    num_segments = math.floor(features["spec"].shape[0]/db_config["len_crop"])
    uttr = []
    content_emb = []
    wav2vec_feat = []
    # crop and append the segments
    for seg_i in range(num_segments):
        uttr.append(features["spec"][seg_i*db_config["len_crop"]:(seg_i+1)*db_config["len_crop"], :])
        content_emb.append(features["phones"][seg_i*db_config["len_crop"]:(seg_i+1)*db_config["len_crop"], :])
        if db_config["speech_input"]=="wav2vec":
            wav2vec_feat.append(features["all_wav2vec_feat"][:, seg_i*db_config["len_crop"]:(seg_i+1)*db_config["len_crop"], :])
    # check if there is a last segment that needs padding
    if((features["spec"].shape[0] - num_segments*db_config["len_crop"]) > 1):
        len_pad = db_config["len_crop"] - (features["spec"].shape[0] - db_config["len_crop"]*num_segments)
        uttr.append(np.pad(features["spec"][num_segments*db_config["len_crop"]:, :], ((0,len_pad),(0,0)), "constant"))
        content_emb.append(np.pad(features["phones"][num_segments*db_config["len_crop"]:, :], ((0,len_pad),(0,0)), "constant"))
        if db_config["speech_input"]=="wav2vec":
            wav2vec_feat.append(np.pad(features["all_wav2vec_feat"][:, num_segments*db_config["len_crop"]:, :], ((0,0), (0,len_pad),(0,0)), "constant"))

    return uttr, content_emb, wav2vec_feat

def no_cut_or_pad_features(features, as_list):
    """ Return features without cutting or padding them """

    wav2vec_feat = features["all_wav2vec_feat"]

    if as_list:
        return [features["spec"]], [features["phones"]], [wav2vec_feat]
    else:
        return features["spec"], features["phones"], wav2vec_feat

def select_data_sample_to_return(db_config, features):
    """ Determines how to return the data sample """

    if db_config["speech_input"]=="wav2vec":
        return features["uttr"], features["emb_org"], features["content_emb"], features["emotion_gt"], features["wav2vec_feat"]
    elif db_config["speech_input"]=="spec":
        return features["uttr"], features["emb_org"], features["content_emb"], features["emotion_gt"]

class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, data_dir, pkl_file, config):
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
            audio = np.load(os.path.join(self.data_dir, element[0]))

            # check if audio and features have the same shape
            check_audio_and_feat_shape(audio=audio,
                                       element=element)

            # define the dataset sample
            dataset[k], new_wav2vec_index = get_dataset_sample(audio=audio,
                                                               element=element,
                                                               config=config)

        self.dataset = list(dataset)
        self.num_tokens = len(self.dataset)
        self.wav2vec_index = new_wav2vec_index

        print("num utterances: ", self.num_tokens) # number of utterances
        print("Finished loading the dataset...")
        
    def __getitem__(self, idx):
        """ Return a data sample (an utterance"s segment). """
        # print("Getting item")
        dataset = self.dataset

        db_config = {}
        db_config["wav2vec_index"] = self.wav2vec_index
        db_config["len_crop"] = self.len_crop
        db_config["wav2vec_path"] = self.wav2vec_path
        db_config["speech_input"] = self.speech_input
        
        # pick a random utterance
        data_sample = dataset[idx]

        # get datasample basic features
        file_name, spec, emb_org, phones, emotion_gt = data_sample[0:5] # audio

        # initialize wav2vec features
        all_wav2vec_feat, wav2vec_feat = init_wav2vec_feat(db_config, file_name)

        features = {}
        features["spec"] = spec
        features["phones"] = phones
        features["emb_org"] = emb_org
        features["all_wav2vec_feat"] = all_wav2vec_feat
        features["emotion_gt"] = emotion_gt

        # print("Cutting and zero padding")
        # if the utterance is too short (zero-pad)
        if spec.shape[0] < self.len_crop:
            uttr, content_emb, wav2vec_feat = zero_pad_feats(features=features,
                                                             db_config=db_config,
                                                             as_list=False)
        # if the utterance is too long (crop)
        elif spec.shape[0] > self.len_crop:
            # randomly crop the utterance
            uttr, content_emb, wav2vec_feat = random_feats_crop(features, db_config)
        # if the utterance has the exact crop size
        else:
            uttr, content_emb, wav2vec_feat = no_cut_or_pad_features(features, as_list=False)
        
        features["uttr"] = uttr
        features["content_emb"] = content_emb
        features["wav2vec_feat"] = wav2vec_feat

        return(select_data_sample_to_return(db_config=db_config,
                                            features=features))
    
    def __len__(self):
        """Return the number of utterances."""
        return self.num_tokens


class EvalUtterances(data.Dataset):
    """ Dataset class to make data loaders for the evaluation.
        This class assumes that the training and prediction occur
        over utterance segments but that the evaluation results
        should be acquired in the utterance level.
    """

    def __init__(self, data_dir, pkl_file, config):
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
            audio = np.load(os.path.join(self.data_dir, element[0]))
            # print("element[0]: ", element[0])
            # print("audio: ", audio.shape)
            # print("element[1]: ", element[1].shape)
            # print("element[2]: ", element[2].shape)
            # print("element[3]: ", element[3])
            # print("element[4]: ", element[4])

            # check if audio and features have the same shape
            check_audio_and_feat_shape(audio=audio,
                                       element=element)

            # define the dataset sample
            dataset[k], new_wav2vec_index = get_dataset_sample(audio=audio,
                                                               element=element,
                                                               config=config)
            
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
        file_name, spec, emb_org, phones, emotion_gt = data_sample[0:5]

        # initialize wav2vec features
        all_wav2vec_feat, wav2vec_feat = init_wav2vec_feat(db_config, file_name)

        features = {}
        features["spec"] = spec
        features["phones"] = phones
        features["emb_org"] = emb_org
        features["all_wav2vec_feat"] = all_wav2vec_feat
        features["emotion_gt"] = emotion_gt

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
    
def get_dataloaders(config, train_pkl, test_pkl, num_workers=4):
    """Build and return a data loader for spectrogram crops (segments) as data samples."""

    train_dir=config.train_dir
    test_dir=config.test_dir
    batch_size=config.batch_size

    # get the training dataset
    train_dataset = Utterances(data_dir=train_dir,
                               pkl_file=train_pkl,
                               config=config)
    
    # get the evaluation datasets (to make utterance-level evaluations from the segments)
    train_eval_dataset = EvalUtterances(data_dir=train_dir,
                                        pkl_file=train_pkl,
                                        config=config)
    test_dataset = EvalUtterances(data_dir=test_dir,
                                  pkl_file=test_pkl,
                                  config=config)
    
    # get the train dataset
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    train_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn,
                                  pin_memory=True)
    
    # get the test dataset cut to perform the utterance-level prediction
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  worker_init_fn=worker_init_fn,
                                  pin_memory=True)

    # get the train dataset cut to perform the utterance-level prediction
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    train_eval = data.DataLoader(dataset=train_eval_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  worker_init_fn=worker_init_fn,
                                  pin_memory=True)

    # training dataset with batch size = 1
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    train_1batch = data.DataLoader(dataset=train_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  worker_init_fn=worker_init_fn,
                                  pin_memory=True)

    return train_loader, test_loader, train_eval, train_1batch
