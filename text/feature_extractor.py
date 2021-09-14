"""
    Name: feature_extractor.py
    
    Description: Extracts text embeddings with the
                 huggingface transformer API.
"""

import glob
import numpy as np
from transformers import AutoModel, AutoTokenizer
import sys
import argparse
import train_helpers as helper

MODEL_NAME = "bert-large-uncased"
FEAT_TYPE = "last_hidden_state"
FOLDER_NAME = "last_hidden_state"
OUT_DIR = "text/" + FOLDER_NAME + "/" + MODEL_NAME

def get_txt_files_list(text_dir, label_dir):
    """ Return the list of text files that have emotion labels
        within the 4 classes (angry, sad, happy or neutral) """
    
    # get the list of all text files
    text_files = glob.glob(text_dir + "/Session*/*/*.txt")

    text_files = helper.filter_text_files(text_files, text_dir, label_dir)
    print("Number of input files after filtering: ", len(text_files))

    return text_files

def get_out_file_name(text_file, text_dir):
    """ Get the path to save the output and its subdir """
    # get output file name
    out_file_name = text_file.split(text_dir + "/")[-1]
    out_file_name = out_file_name[:-4] + ".npy"

    # get subdir
    subdirs = out_file_name.split("/")
    subdir = ""
    for i in range(len(subdirs)-1):
        subdir += subdirs[i] + "/"

    return out_file_name, subdir

def extract_features(text_files, text_dir, text_model, tokenizer):
    """ Extract features for all the text files """
    # for all the text files
    for text_file in text_files:
        # read text
        text, success = helper.read_text_file(text_file)
        if success:
            # tokenize the texts
            text = text.lower()
            text = text.replace("\n", "")
            encoded_input = tokenizer(text, padding="max_length", max_length=122, return_tensors='pt')

            # get the model's output
            output = text_model(**encoded_input)

            # get the text embedding
            if(FEAT_TYPE=="last_hidden_state"):
                semantic_feat = output.last_hidden_state
            else:
                print("Error! Non existing feature type ", FEAT_TYPE)
                sys.exit(1)

            # transform tensor to numpy
            np_tensor = semantic_feat.detach().numpy()

            # get the output file name and its subdir
            out_file_name, subdir = get_out_file_name(text_file, text_dir)
            
            # create the output subdir
            helper.create_dir(OUT_DIR + "/" + subdir)

            # save the numpy embedding in OUT_DIR
            np.save(OUT_DIR + "/" + out_file_name, np_tensor)
            print("Saved ", OUT_DIR + "/" + out_file_name)

            encoded_text = tokenizer.tokenize(text, return_tensors='pt')
            helper.create_dir(OUT_DIR + "/tokenization_result/" + subdir)
            helper.save_tokenization(encoded_text, OUT_DIR + "/tokenization_result/" + out_file_name[:-4] + ".txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_dir", type=str, help="directory to text files")
    parser.add_argument("--label_dir", type=str, help="directory to label files")

    config = parser.parse_args()
    text_dir = config.text_dir
    label_dir = config.label_dir

    # We need to create the feature extractor model and tokenizer
    text_model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # get list of files
    text_files = get_txt_files_list(text_dir=text_dir,
                                    label_dir=label_dir)

    print("Number of text files: ", len(text_files))
    
    # extract text features
    if(len(text_files) > 0):
        extract_features(text_files=text_files,
                        text_dir=text_dir,
                        text_model=text_model,
                        tokenizer=tokenizer)

if __name__ == '__main__':
    main()
