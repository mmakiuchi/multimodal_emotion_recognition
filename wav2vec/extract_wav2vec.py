"""
    Name: eval_and_extract_wav2vec.py

    Description: Script to extract wav2vec embeddings from the model's
                 last hidden layer and store these embeddings as npy
"""

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import soundfile as sf
import torch
import sys, os
import numpy as np
from spectrogram_helpers import get_spec

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print("Device: ", device)

MODEL_TYPE = "facebook/wav2vec2-large-lv60"

processor = Wav2Vec2Processor.from_pretrained(MODEL_TYPE)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_TYPE)

# model to cuda
model = model.to(device)

def map_to_array(batch):
    """ Read the speech files and store the raw speech in batch """
    
    # read the speech file
    speech, _ = sf.read(batch["file"])

    # define padding to match the spectrograms
    if(len(speech)%320 >= 80):
        padding = 160
    else:
        padding = 319

    # pad the speech data
    speech = np.pad(speech, padding, mode='constant')
    
    batch["speech"] = speech
    return batch

def save_hidden_states(speech_files, out_dir, hidden_states, speech):
    """ Save wav2vec hidden states as npy files """
    for speech_file, sp in zip(speech_files, speech):
        file_name = speech_file.split("/")[-1]
        file_name = file_name[:-4] + ".npy"
        subdir1 = speech_file.split("/")[-2]
        subdir2 = speech_file.split("/")[-3]
        
        # get the spectrogram and compare the shape of the spectrogram and the
        # shape of the wav2vec features
        spec = get_spec(speech_file)

        if not (spec.shape[0] == hidden_states[-1].detach().cpu().numpy().shape[1]):
            print("speech file: ", speech_file)
            print("speech len: ", len(sp))
            print("spec: ", spec.shape)
            print("features: ", hidden_states[-1].detach().cpu().numpy().shape)
            sys.exit(1)
        
        # make a list of numpy arrays
        tmp_states = []
        for element in hidden_states:
            # transform the tensors to numpy and append to the list
            tmp_states.append(element.detach().cpu().numpy())

        # transform the list of numpy into numpy
        tmp_states = np.concatenate(tmp_states, axis=0)

        # save all hidden states as torch
        out_file = out_dir + "/all_hidden_states/" + subdir2 + "/" + subdir1 + "/" + file_name

        if not os.path.exists(out_dir + "/all_hidden_states/" + subdir2 + "/" + subdir1):
            os.makedirs(out_dir + "/all_hidden_states/" + subdir2 + "/" + subdir1)

        np.save(out_file, np.array(tmp_states))


def map_to_pred(batch):
    inputs = processor(batch["speech"], return_tensors="pt", sampling_rate=16000, padding=True)
    
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    out_dir = "wav2vec/features/large-lv60"

    # we run the model in inference mode just to get the output
    with torch.no_grad():
        output = model(input_values, attention_mask=attention_mask, output_hidden_states=True)

    save_hidden_states(speech_files=batch["file"],
                       out_dir=out_dir,
                       hidden_states=output.hidden_states,
                       speech=batch["speech"])
    
    ## get the text results from the decoding
    # logits = output.logits
    # predicted_ids = torch.argmax(logits, dim=-1)
    # transcription = tokenizer.batch_decode(predicted_ids)
    # transcription = processor.batch_decode(predicted_ids)
    # batch["transcription"] = transcription

def eval(data_file):
    """ Evaluate the model on the librispeech dataset, printing the WER """
    # load the dataset
    librispeech_eval = load_dataset("csv", data_files=data_file, column_names=["file","text"], delimiter=";", quoting=3, split="train")
    librispeech_eval = librispeech_eval.map(map_to_array)
    librispeech_eval.map(map_to_pred, batched=True, batch_size=1, remove_columns=["speech"])

def main():
    # samples that have text transcript
    data_file = "wav2vec/utils/all_sessions.csv"
    eval(data_file)

    # samples that have no text transcript
    no_content_files = "wav2vec/utils/no_content_files.csv"
    eval(no_content_files)

if __name__ == '__main__':
    main()