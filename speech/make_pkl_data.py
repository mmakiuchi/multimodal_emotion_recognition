"""
    Name: make_pkl_data.py
    Description: Generate speaker embeddings and organize the
                 data in a pkl file.
"""

import os, sys
import numpy as np
from pathlib import Path

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import speech.make_data_helper as helper

# how many utterances to average to compute the speaker id embedding
num_uttrs = 100

def get_data(config):
    """ Main loop to get the data for the IEMOCAP dataset """

    dir_name, subdir_list, _ = next(os.walk(config.root_dir))
    helper.logger("info", "[INFO] Found directory: " + str(dir_name))

    utterances = []
    all_success = 0
    all_unsuccess = 0

    # for all sessions
    for session in sorted(subdir_list):
        helper.logger("info", "[INFO] Processing session: " + str(session))

        session_dir, spks_gender, _ = next(os.walk(os.path.join(dir_name,session)))
        
        # for both speakers (Male and Female)
        for gender in spks_gender:
            phone_seq_file_path = config.phone_dir + "/" + session + "/" + gender # path to the speaker json alignment files
            pathlib_phone_path = Path(phone_seq_file_path)
            
            # if there are json alignment files for the current speaker
            if(pathlib_phone_path.exists()):
                #################################################
                #           1. Get the speaker id               #
                #################################################
                spk_id_to_append = session + "_" + gender
                print("spk: ", spk_id_to_append)
                # get the utterance files
                _ , _, file_list = next(os.walk(os.path.join(session_dir,gender)))

                #################################################
                #          2. Get speaker embedding             #
                #################################################
                assert len(file_list) >= num_uttrs
                helper.logger("info", "[INFO] len(file_list): " + str(len(file_list)))

                # get utterance indices list in a random order
                idx_uttrs = np.random.choice(len(file_list), size=num_uttrs, replace=False)
                helper.logger("info", "[INFO] idx_uttrs: " + str(idx_uttrs))
                embs = helper.get_speaker_embeddings(dir_name=config.wav_dir,
                                                    speaker=session + "/" + gender,
                                                    file_list=file_list,
                                                    idx_uttrs=idx_uttrs,
                                                    num_uttrs=num_uttrs)
                
                spk_emb_to_append = np.mean(embs, axis=0)
                helper.logger("info", "[INFO] Speaker embedding mean shape: " + str(np.mean(embs, axis=0).shape))

                #################################################
                #           3. Make content embedding           #
                #################################################
                content_list = []
                content_list, success_cases, unsuccess_cases = helper.get_content_list(file_list=file_list,
                                                                                       speaker_dir=session + "/" + gender,
                                                                                       config=config)
                all_success += success_cases
                all_unsuccess += unsuccess_cases
                
                if(len(content_list) > 0):
                    for element in content_list:
                        tmp_element = element.copy()
                        if(gender == "Female"):
                            gender_class = 0
                        elif(gender == "Male"):
                            gender_class = 1
                        else:
                            print("Error. Undefined gender " + str(gender))
                            sys.exit(1)
                        tmp_element.append(gender_class)
                        tmp_element.append(spk_emb_to_append)
                        tmp_element.append(spk_id_to_append)
                        utterances.append(tmp_element)
                        # helper.logger("info", "[INFO] Utterances length: " + str(len(utterances)))
            else:
                helper.logger("warning", "[WARNING] The speaker does not have any phone sequence")
        
    helper.save_data(utterances, config)
    helper.logger("info", "[INFO] Total number of successful spec transformations: " + str(all_success))
    helper.logger("info", "[INFO] Total number of unsuccessful spec transformations: " + str(all_unsuccess))

def main():

    config = helper.get_config()
    get_data(config)

if __name__ == '__main__':
    main()