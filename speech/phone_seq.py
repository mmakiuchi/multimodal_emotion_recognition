"""
    Name: phone_seq.py
    Description: loads the phone sequences stored as json files
"""

import csv, sys, json, math
import numpy as np

import make_data_helper as helper

from pathlib import Path

def read_json(json_file):
    """ Reads the json file and returns its contents. """
    with open(json_file) as json_pointer:
        json_data = json.load(json_pointer)

    return json_data

def get_phone_dict(dict_file):
    """ Returns the index dictionary file for all the phone types """
    phone_dict = {}
    with open(dict_file, mode="r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            phone_dict[row[0]] = int(row[1])

    return phone_dict

def phones2onehot(phones, num_labels):
    """ Returns the one_hot version of the phones list """
    one_hot_phones = []

    for phone in phones:
        onehot_encoded = np.zeros(num_labels)
        onehot_encoded[phone] = 1
        onehot_encoded = onehot_encoded.astype(np.single)
        one_hot_phones.append(onehot_encoded)

    return one_hot_phones

def get_word_seq_and_intervals(json_file):
    """ Return the word sequence and their intervals (tuple (begin, end) in seconds) """

    word_seq = []
    word_intervals = []

    # get json data
    json_data = read_json(json_file)

    if not ("words" in json_data):
        helper.logger("warning", "[WARNING] There is no keyword -words- in the json file " + str(json_file))
        return [], []

    # go through all words
    words = json_data["words"]
    for word in words:
        if (word["case"] == "success"):
            word_start = float(word["start"])
            word_end = float(word["end"])
            word_seq.append(word["word"].lower()) # append lower-case word
            word_intervals.append([word_start, word_end])

    assert len(word_seq) == len(word_intervals)

    return word_seq, word_intervals

def remove_punctuation(word):
    """ Remove punctuation from string word """
    new_word = word.replace("-", "")
    new_word = new_word.replace(".", "")
    new_word = new_word.replace(",", "")
    new_word = new_word.replace("!", "")
    new_word = new_word.replace("?", "")

    return new_word

def get_token_id(txt_file, post_word, is_eof, phone_dict):
    """ Return the token id (token or silence)
        that exist before post_word.
    """
    if(len(post_word) > 0):
        post_word = remove_punctuation(post_word)
    txt_f_pt = Path(txt_file)
    if txt_f_pt.exists():
        with open(txt_file, "r") as txt_p:
            line = txt_p.readline()
            line = line.lower()
            line = remove_punctuation(line)

        if is_eof:
            # check if the line ends with a token
            new_line = line.replace(" ", "")
            new_line = new_line.replace("\n","")
            # if the last character is the end of token
            if(new_line[-1] == "]"):
                count = 0
                for i in reversed(new_line):
                    if(i == "["):
                        break
                    count -=1
                token_id = phone_dict[new_line[count:-1].upper()]
            else:
                token_id = 0
        else:
            # check if the previous word was a token
            words = line.split(" ")
            prev_word = None
            for word in words:
                if(post_word==word):
                    break
                prev_word = word
            if(prev_word == None):
                token_id = 0
            else:
                if("[" in prev_word) or ("]" in prev_word):
                    prev_word = prev_word.replace(" ", "")
                    prev_word = prev_word.replace("\n","")
                    token_id = phone_dict[prev_word[prev_word.find("[")+1:-1].upper()]
                else:
                    token_id = 0
        return token_id
    else:
        return 0

def get_txt_file(config, speaker_dir, file_name):
    """ Return the txt file. """
    return config.txt_dir + '/' + speaker_dir + '/' + file_name[:-4] + '.txt'

def get_phone_info_from_json(json_file, config, spec_frames, speaker_dir, file_name):
    """ Return the phones and their durations read from the json file """

    phone_dict = config.phone_dict
    frame_len = config.frame_len

    spec_duration = float(spec_frames * frame_len) # spectrogram duration in seconds

    # read json file
    json_data = read_json(json_file=json_file)

    # get corresponding text file path
    txt_file = get_txt_file(config, speaker_dir, file_name)

    # check if there are words in the json file
    if not ("words" in json_data):
        helper.logger("error", "[ERROR] There is no keyword -words- in the json file " + str(json_file))
        return [], False

    # get list of words from the json file
    words = json_data["words"]
    phones_and_durations = [] # list with the phones and their durations

    accum_dur = 0.0
    appended_duration = 0.0
    is_prev_word_unk = False

    for word in words:
        condition = (word["case"] == "success") or (word["case"] == "not-found-in-audio")
        MODEL_TOKENS = True # model silence and tokens between words
        if not condition:
            helper.logger("warning", "[WARNING] Not success. Case is " + str(word["case"]) + " for json file " + str(json_file))
            return [], False

        if (word["case"] == "success"):
            word_start = float(word["start"])
            phones = word["phones"]
            if (MODEL_TOKENS and (accum_dur < word_start)):
                if(float(word_start-accum_dur) >= float(frame_len/10)):
                    if is_prev_word_unk:
                        token_id = 1 # unknown word id
                    else:
                        token_id = get_token_id(txt_file, word["word"].lower(), False, phone_dict)
                    phones_and_durations.append([token_id, word_start-accum_dur]) # append token
                    appended_duration += (word_start-accum_dur)
            
            accum_dur = word_start
            for phone in phones:
                if(phone["phone"]=="sil"):
                    if(float(phone["duration"]) >= float(frame_len/10)):
                        phones_and_durations.append([int(0), float(phone["duration"])])
                        appended_duration += float(phone["duration"])
                else:
                    if(float(phone["duration"]) >= float(frame_len/10)):
                        phones_and_durations.append([int(phone_dict[phone["phone"]]), float(phone["duration"])])
                        appended_duration += float(phone["duration"])
                accum_dur += phone["duration"]
            is_prev_word_unk = False
        else:
            # word not-found-in-audio
            is_prev_word_unk = True

    # add the token at the end of the sequence
    if((accum_dur < spec_duration) and MODEL_TOKENS):
        token_duration = spec_duration-accum_dur
        if(float(token_duration) >= float(frame_len/10)):
            if is_prev_word_unk:
                token_id = 1 # unknown word id
            else:
                token_id = get_token_id(txt_file, [], True, phone_dict)
            phones_and_durations.append([token_id, float(token_duration)]) # append token at the end
            appended_duration += float(token_duration)

    return phones_and_durations, True

def get_phone_seq(json_file, config, spec_frames, speaker_dir, file_name):
    """ Returns the phone sequence according to the mode from the json file """
    
    phone_dict = config.phone_dict
    frame_len = config.frame_len

    # get the phones and durations from the json file
    phones_and_durations, success = get_phone_info_from_json(json_file, config, spec_frames, speaker_dir, file_name)

    if not success:
        return [], [], False

    total_phones_dur = 0.0 # store the accumulated duration of all the read phones
    time_main_phones = 0.0 # accumulated time of phones in the main phones list
    prev_phones = [] # store previous phones that had incomplete frames
    main_phones = [] # list with the main phones of each spec frame

    # for all phones and their durations
    for element in phones_and_durations:
        phone = element[0]
        phone_duration = element[1]
        appended_frame = False

        # how much time the pointer of the main phone list is ahead of the phone duration list reading
        difference = round(time_main_phones - total_phones_dur, 4)
        if(difference >= frame_len):
            helper.logger("error", "[ERROR] Something went wrong. A complete frame was not appended in the past")
            sys.exit(1)
        
        if((difference <= -frame_len) and (len(prev_phones) > 0)):
            # append the previous phone with the largest contribution
            largest_prev_duration = 0.0
            for prev_phone in prev_phones:
                if(prev_phone[1] > largest_prev_duration):
                    largest_prev_duration = prev_phone[1]
                    main_prev_phone = prev_phone[0]
            main_phones.append(main_prev_phone)
            time_main_phones += largest_prev_duration
            if(difference - frame_len > 0):
                last_phone = prev_phones[-1]
                prev_phones = []
                prev_phones.append(last_phone)
            else:
                prev_phones = []
            difference = time_main_phones - total_phones_dur # update the difference

        relative_duration = phone_duration - difference
        if(relative_duration >= frame_len/2):
            complete_frames = int(math.floor(relative_duration/frame_len))
            
            # append complete frames
            if(complete_frames > 0):
                for i_frames in range(complete_frames):
                    main_phones.append(phone)
                    appended_frame= True
                time_main_phones += complete_frames*frame_len
                prev_phones = []
                if((relative_duration - complete_frames*frame_len) > frame_len/10):
                    prev_phones.append((element[0], relative_duration - complete_frames*frame_len))
            
            # the current phone is the main phone of the last frame
            if((relative_duration - (complete_frames*frame_len)) >= frame_len/2):
                main_phones.append(phone)
                time_main_phones += frame_len
                appended_frame = True
                prev_phones = [] # clear the list

            # if no frame was appended in this iteration of the loop
            if(appended_frame==False):
                prev_phones.append(element) # append the phone and its duration
        else:
            if(appended_frame==False):
                prev_phones.append(element) # append the phone and its duration

        # accumulated time
        total_phones_dur = total_phones_dur + phone_duration

    # cropping or adding elements to the main phone sequence
    if((len(main_phones) < spec_frames) and (len(prev_phones) > 0)):
        largest_prev_duration = 0.0
        sum_prev_len= 0.0
        for prev_phone in prev_phones:
            if(prev_phone[1] > largest_prev_duration):
                largest_prev_duration = prev_phone[1]
                main_prev_phone = prev_phone[0]
            sum_prev_len += prev_phone[1]
        main_phones.append(main_prev_phone)
    elif(len(main_phones) == (spec_frames + 1)):
        # if the difference is only one, it was probably an approximation error
        main_phones = main_phones[:spec_frames]
    elif(len(main_phones) > spec_frames + 1):
        print("[ERROR] Something went wrong. There are more main phones than spectrogram frames")
    elif((len(main_phones) < spec_frames) and ((spec_frames - len(main_phones)) <=1)):
        # repeat the last main_phone at the end at most twice
        for p_i in range(spec_frames - len(main_phones)):
            main_phones.append(main_phones[-1])

    assert len(main_phones) == spec_frames
    
    main_phones = phones2onehot(phones=main_phones,
                                    num_labels=len(phone_dict)+2) # +2 to model silence and unknown

    return np.array(phones_and_durations), np.array(main_phones), success

def get_silent_phone_seq(config, spec_frames, speaker_dir, file_name):
    """ Return a silent phone sequence. """

    frame_len = config.frame_len
    phone_dict = config.phone_dict

    spec_duration = float(spec_frames * frame_len) # spectrogram duration in seconds

    txt_file = get_txt_file(config, speaker_dir, file_name)
    txt_f_pt = Path(txt_file)
    # if the file exists
    if txt_f_pt.exists():
        with open(txt_file, "r") as txt_p:
            line = txt_p.readline()
        if(len(line) > 0):
            line = line.replace(" ", "")
            line = line.replace("\n","")
            token = phone_dict[line[1:-1].upper()]
        else:
            token = 0 # silence
    # if there is no transcript file, it is silence
    else:
        token = 0

    # the whole duration is silence
    phones_and_durations = []
    phones_and_durations.append([token, spec_duration])
    main_phones = []
    for i in range(spec_frames):
        main_phones.append(token)

    main_phones = phones2onehot(phones=main_phones,
                                num_labels=len(phone_dict)+2) # +2 to model silence and unknown
    
    assert len(main_phones) == spec_frames
    return np.array(phones_and_durations), np.array(main_phones)