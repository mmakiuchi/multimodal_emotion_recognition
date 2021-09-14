"""
    Name: extract_phonemes.py
    
    Description: Script to extract the phoneme sequence using Gentle.
"""

import os, glob, argparse, random, sys
from pathlib import Path

def create_dir(save_path):
    """ Test if the directory exists and, if it does not, it creates it """

    path = Path(save_path)

    if not path.exists():
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            print("[ERROR] It was not possible to create the directory " + str(save_path))
            sys.exit(3)

def filter_txt_files(txt_fs, json_fs, txt_dir):
    """ Returns the txt files that were still not converted to json files """
    
    filtered_txt_fs = []

    for txt_f in txt_fs:
        has_json=False
        splits = txt_f.split(txt_dir + "/")
        splits = splits[1].split("/")
        txt_name = splits[-1]
        f_name = txt_name[:-4]

        for json_f in json_fs:
            if(f_name in json_f):
                has_json = True
                break

        if(has_json==False):
            filtered_txt_fs.append(txt_f)

    random.shuffle(filtered_txt_fs)

    return filtered_txt_fs

def filter_script_improv(file_list, scripted):
    """
        Returns the list with only the scripted
        files if scripted=True or with only the improvised
        files if scripted=False
    """
    filtered_list = []
    for path in file_list:
        if(scripted):
            if("_script" in str(path)):
                filtered_list.append(path)
        else:
            if("_impro" in str(path)):
                filtered_list.append(path)

    return filtered_list

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Speaker id (to get the alignment for few speaker at time):
    # M = Male; F = Female
    # 1,2,3,4,5 = session number
    # i = improvised; s = scripted
    ap.add_argument("--speaker", type=str, required=True,
                    help="Speaker id (e.g., M1i, M1s, F2i, F2s)")
    ap.add_argument("--txt_dir", type=str, required=True,
                    help="Directory for the text files")
    ap.add_argument("--audio_dir", type=str, required=True,
                    help="Directory for the speech files")
    config = ap.parse_args()

    # get the configuration variables
    speaker = config.speaker
    txt_dir = config.txt_dir
    audio_dir = config.audio_dir

    print("Speaker: ", speaker)
    txt_ext = ".txt"
    audio_ext = ".wav"
    output_main = "align_results"

    if("F" in str(speaker)):
        gender = "Female"
    elif("M" in str(speaker)):
        gender = "Male"
    else:
        print("Invalid gender for speaker " + str(speaker))
        sys.exit(1)

    if("1" in str(speaker)):
        session = "Session1"
    elif("2" in str(speaker)):
        session = "Session2"
    elif("3" in str(speaker)):
        session = "Session3"
    elif("4" in str(speaker)):
        session = "Session4"
    elif("5" in str(speaker)):
        session = "Session5"  
    else:
        print("Invalid session number for speaker " + str(speaker))
        sys.exit(1)
    
    if("s" in str(speaker)):
        scripted = True
    elif("i" in str(speaker)):
        scripted = False
    else:
        print("Invalid scripted/improvised option for speaker " + str(speaker))
        sys.exit(1)

    txt_fs = glob.glob(txt_dir + "/" + str(session) + "/" + str(gender) + "/*.txt")
    json_fs = glob.glob(output_main + "/" + str(session) + "/" + str(gender) + "/*.json")
    txt_fs = filter_script_improv(file_list=txt_fs,
                                    scripted=scripted)
    if(len(json_fs) > 0):
        json_fs = filter_script_improv(file_list=json_fs,
                                       scripted=scripted)
    print("Number of text files after filtering script/impro: ", len(txt_fs))
    print("Number of json files so far after filtering script/impro: ", len(json_fs))

    # filter according to the remaining text files
    filtered_txt_fs = txt_fs
    if(len(json_fs) > 0):
        filtered_txt_fs = filter_txt_files(txt_fs=txt_fs,
                                           json_fs=json_fs,
                                           txt_dir=txt_dir)

        if not (len(filtered_txt_fs) == (len(txt_fs)-len(json_fs))):
            print("Error. The number of remaining txt files to transform is invalid and equal to ", len(filtered_txt_fs))
            sys.exit(1)
        print("Number of remaining text files: ", len(filtered_txt_fs))
    
    for txt_f in filtered_txt_fs:
        splits = txt_f.split(txt_dir + "/")
        splits = splits[1].split("/")
        folders = splits[:-1]
        folder_name = []
        for folder in folders:
            if(folder_name==[]):
                folder_name = folder
            else:
                folder_name = folder_name + "/" + folder
        txt_name = splits[-1]
        f_name = txt_name[:-4]
        audio_f = audio_dir + "/" + folder_name + "/" + f_name + audio_ext
        create_dir(output_main + "/" + folder_name)
        output_f = output_main + "/" + folder_name + "/" + f_name + ".json"

        audio_path = Path(audio_f)
        if audio_path.exists():
            os.system("python3 align.py -o " + str(output_f) + " " + str(audio_f) + " " + str(txt_f))
        else:
            print("The audio path " + str(audio_f) + " does not exist")