"""
    Name: make_data_helper.py
    Description: helper functions to store the data in a pkl file.
"""

import os, csv
import pickle
import phone_seq as ph
import numpy as np
from pathlib import Path
import logging # to log
from termcolor import colored # to make the text colorful
from datetime import datetime
import argparse

# to get the speaker embeddings
from resemblyzer import preprocess_wav, VoiceEncoder
from tqdm import tqdm

spk_encoder = VoiceEncoder()

now = datetime.now()
date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
LOG_PATH = "make_pkl_data_" + date_time + ".log"

def get_config():
    """ Get paths and other configurations """
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, help="training or test mode")
    parser.add_argument("--spmel_dir", type=str, help="string to define the directory")
    parser.add_argument("--dict_file", type=str, help="path to the phone dictionary file")
    parser.add_argument("--labels_dir", type=str, help="path to the label files")
    parser.add_argument("--wav_dir", type=str, help="path to the speech wave files")
    parser.add_argument("--txt_dir", type=str, help="path to the text files")
    parser.add_argument("--phone_dir", type=str, help="path to the phonetic alignment json files")
    parser.add_argument("--freq", type=int, default=16000, help="speech frequency")
    parser.add_argument("--hop_len", type=int, default=320, help="hop length")

    parser_config = parser.parse_args()
    
    phone_dict = ph.get_phone_dict(parser_config.dict_file)
    
    parser_config.frame_len = float(parser_config.hop_len/parser_config.freq) # = hop_length/freq spectrogram frame duration in seconds
    parser_config.phone_dict = phone_dict
    
    parser_config.root_dir = "./" + str(parser_config.spmel_dir)

    print(parser_config)
    return parser_config

def logger(level_name, message, log_path=LOG_PATH, highlight=False, show_terminal=True):
    """
        Write message to the log
        Input:
                level_name : level name (e.g., error, warning, etc)
                message    : message to be printed in the screen (prompt) and log
                log_path   : path of the log file
                highlight  : boolean indicating if the message should be highlighted in the prompt
        Output: void
    """

    # log configuration
    logging.basicConfig(filename=log_path, filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')

    if level_name == 'info':
        logging.info(message)
        if show_terminal:
            if highlight:
                print(colored(message,'white', 'on_green'))
            else:
                print(colored(message,'green'))
    elif level_name == 'error':
        logging.error(message)
        if show_terminal:
            print(colored(message,'red'))
    elif level_name == 'warning':
        logging.warning(message)
        if show_terminal:
            print(colored(message,'yellow'))

def save_data(speakers, config):
    """ Save pkl data file """
    logger('info','[INFO] Speakers length: ' + str(len(speakers)))

    out_f_name = config.mode + '.pkl'
    with open(os.path.join(config.root_dir, out_f_name), 'wb') as handle:
        # serialize the object "speakers" and store in the file "handle"
        pickle.dump(speakers, handle)

def get_emotion_class(speaker_dir, file_name, config):
    """ Return the emotion class given an utterance file """

    emotion_class = 0
    
    label_path = config.labels_dir + "/" + speaker_dir + "/" + file_name[:-4] + ".csv"
    with open(label_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = row["category"]
            break
    if((label=="Happy") or (label=="Excited")):
        emotion_class=2
    elif(label=="Angry"):
        emotion_class=0
    elif(label=="Neutral"):
        emotion_class=1
    elif(label=="Sad"):
        emotion_class=3
    return emotion_class

def is_file_unk_token(txt_f):
    """ Identify if the txt file is only an unknown token. """
    # read the txt file
    with open(txt_f, "r") as txt_p:
        transcript = txt_p.readline()
    transcript = transcript.strip()
    if((transcript[0] == "[") and (transcript[-1] == "]")):
        return True
    else:
        return False

def get_content_list(file_list, speaker_dir, config):
    """
        Get the content list for a given speaker
        The content list includes the utterance file path,
        the sequence of phones and durations, the sequence
        of main phones and the emotion for each utterance
    """
    content_list = []
    unsuccess_cases = 0
    success_cases = 0

    # for all files of a given speaker
    for file_name in sorted(file_list):
        print("Processing ", file_name)
        json_file_name = file_name[:-4] + '.json'
        phone_seq_file_path = config.phone_dir + '/' + speaker_dir + '/' + json_file_name
        pathlib_phone_path = Path(phone_seq_file_path)

        spec = np.load(str(config.root_dir) + '/' + speaker_dir + '/' + file_name) # load spectrogram

        if pathlib_phone_path.exists():
            # The speech has phonetic content
            phones_and_durations, main_phones, success = ph.get_phone_seq(json_file=phone_seq_file_path,
                                                                          config=config,
                                                                          spec_frames=spec.shape[0],
                                                                          speaker_dir=speaker_dir,
                                                                          file_name=file_name)
            if success:
                assert main_phones.shape[0] == spec.shape[0]
                utt_file = os.path.join(speaker_dir,file_name)
                emotion_class = get_emotion_class(speaker_dir, file_name, config)
                word_seq, word_intervals = ph.get_word_seq_and_intervals(json_file=phone_seq_file_path)
                utt_content = [str(utt_file), phones_and_durations, main_phones, emotion_class, word_seq, word_intervals] # save spmel file name and phone sequence
                content_list.append(utt_content)
                success_cases += 1
            else:
                unsuccess_cases += 1
        else:
            # The speech corresponds to a "silent" speech get the transcript file
            txt_f = config.txt_dir + '/' + speaker_dir + '/' + file_name[:-4] + '.txt'
            txt_f_pt = Path(txt_f)
            if txt_f_pt.exists():
                # if the file only has unknown tokens (no phone information)
                if is_file_unk_token(txt_f):
                    phones_and_durations, main_phones = ph.get_silent_phone_seq(config=config,
                                                                               spec_frames=spec.shape[0],
                                                                               speaker_dir=speaker_dir,
                                                                               file_name=file_name)
                    assert main_phones.shape[0] == spec.shape[0]
                    utt_file = os.path.join(speaker_dir,file_name)
                    emotion_class = get_emotion_class(speaker_dir, file_name, config)
                    utt_content = [str(utt_file), phones_and_durations, main_phones, emotion_class, [None], [None]] # save spmel file name and phone sequence
                    content_list.append(utt_content)
                    success_cases += 1
                else:
                    logger("warning", "[WARNING] File " + str(txt_f) + " was not aligned properly!")
                    unsuccess_cases =+ 1
            else:
                logger("warning", "[WARNING] Unexisting path " + str(pathlib_phone_path) + " or " + str(txt_f))
                unsuccess_cases =+ 1

    logger("info", "[INFO] Number of successful specs: " + str(success_cases))
    logger("info", "[INFO] Number of unsuccessful specs: " + str(unsuccess_cases))
    return content_list, success_cases, unsuccess_cases

def get_speaker_embeddings(dir_name, speaker, file_list, idx_uttrs, num_uttrs):
    """ Return speaker embeddings from a speaker encoder """
    
    # make list with num_uttrs random wave files
    wav_files = []
    for i in range(num_uttrs):
        file_name = file_list[idx_uttrs[i]][:-4] + ".wav"
        wav_file = str(dir_name) + "/" + str(speaker) + "/" + str(file_name)
        wav_files.append(wav_file)

    speaker_wavs = np.array(list(map(preprocess_wav, tqdm(wav_files, "Preprocessing wavs", len(wav_files)))))
    utterance_embeds = np.array(list(map(spk_encoder.embed_utterance, speaker_wavs)))

    return utterance_embeds