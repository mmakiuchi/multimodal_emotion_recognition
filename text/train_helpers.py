"""
    Name: train_helpers.py
    
    Description: Helper functions to train text models
                 with text embeddings extracted using
                 the transformers huggingface API
"""

import csv
from pathlib import Path

def save_tokenization(encoded_text, file_path):
    """ Save the tokenization result in the file path"""
    with open(file_path, "w") as fp:
        for token in encoded_text:
            fp.write(token + "\n")

def create_dir(file_path):
    """ Creates a directory """
    file_path = Path(file_path)

    if not file_path.exists():
        file_path.mkdir(parents=True)

def read_label_csv(label_path):
    """ Reads the label csv file and returns the label category. """
    with open(label_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label_cat = row["category"]
            break
    if(label_cat == "Excited"):
        label_cat = "Happy"
    return label_cat

def get_label_file(input_file, input_dir, label_dir):
  """ Return the label file for a given input file. """
  subdirs = input_file.split(input_dir + "/")[-1]
  return label_dir + "/" + subdirs[:-4] + ".csv"

def filter_text_files(text_files, text_dir, label_dir):
  """ Filter the text files to return only the files with the 5 classes """
  classes = ["Angry", "Happy", "Excited", "Sad", "Neutral"]

  filtered_files = []
  for text_file in text_files:
    # get the label file
    label_path = get_label_file(text_file, text_dir, label_dir)

    # read the label file
    pathlib_label_path = Path(label_path)
    if(pathlib_label_path.exists()):
      label_cat = read_label_csv(label_path)
      if label_cat in classes:
        filtered_files.append(text_file)

  return filtered_files

def read_text_file(text_file):
    """ Reads a text file composed of a single line """
    text = []
    with open(text_file, "r") as tp:
        text = tp.readline()
    if(text[0] == "["):
        return [], False
    text.replace("\n", "")
    return text, True


def get_label_file_path(root_dir, label_dir, npy_file):
    """ Return the path to the label file """
    subdir = npy_file.split(root_dir + "/")[1]
    label_path = label_dir + "/" + subdir[:-4] + ".csv"

    return label_path

def test_is_4_classes(label_cat, emo_categories):
    """ Tests if the label belongs to the 4 emotion classes. """
    # gets the emotion labels of the 4 classes
    keys = emo_categories.keys()
    if label_cat in keys:
        return True
    else:
        return False

def get_emotion_label(root_dir, label_dir, npy_file, emo_categories):
    """ Returns the emotion label for a given json file. """
    label_path = get_label_file_path(root_dir, label_dir, npy_file)
    label_cat = read_label_csv(label_path)
    is_4classes = test_is_4_classes(label_cat, emo_categories)
    if is_4classes:
        emo_label = emo_categories[label_cat]
        return emo_label, True
    else:
        return -1, False

def get_class_recall_dict(num_classes):
    """ Return the class recall dictionary
        This dictionary stores the true positive and
        the positive values for all the emotion classes. """
    
    class_recall = {}
    for class_i in range(num_classes):
        class_recall[str(class_i)] = {}
        class_recall[str(class_i)]["tp"] = 0
        class_recall[str(class_i)]["p"] = 0
    return class_recall
