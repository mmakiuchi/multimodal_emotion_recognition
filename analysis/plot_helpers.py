"""
    Name: plot_helpers.py
    
    Description: Scripts to get UMAP and LDA plots
"""

from umap import UMAP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
import sys, csv, pickle, os
import torch
from math import ceil

# we define all the emotion classes in iemocap, but we only use the classes 1 to 5,
# and "Happy" and "Excited" are merged in a single class
emotion_classes = {
    "Angry":1,
    "Neutral":2,
    "Happy":3,
    "Excited":4,
    "Sad":5,
    "Disgust":6,
    "Frustrated":7,
    "Fear":8,
    "Surprise":9,
    "Other":10,
    "XXX":11
}

# colors used in the plots
_my_colors = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=np.float) / 255 

################ Plot Functions ################
def plot_umap(embeds, labels, markers=None, legend=True, 
              title="", legend_title="Labels", fig_name="umap.eps", **kwargs):
    """ Plot a UMAP embedding plot for embeds according to labels """
    
    # define the plot
    _, ax = plt.subplots(figsize=(10, 7))
        
    # compute the 2D projections with umap
    reducer = UMAP(**kwargs)
    projs = reducer.fit_transform(embeds, labels)
    
    labels = np.array(labels)

    # draw the projections
    for i, speaker in enumerate(np.unique(labels)):
        speaker_projs = projs[labels == speaker]
        marker = "+" if markers is None else markers[i]
        label = speaker if legend else None
        ax.scatter(*speaker_projs.T, c=[_my_colors[i]], marker=marker, label=label)

    # draw the legend
    if legend:
        ax.legend(title=legend_title, ncol=2)
    
    # define title, xticks, yticks
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # save the umap plot as an eps file
    plt.savefig(fig_name, format='eps', pad_inches=0)

def plot_lda(embs, labels, title, fig_name):
    """ Plot the LDA projections """

    # get the 2D projections with LDA
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(embs, labels).transform(embs)

    # define the plot figure
    plt.figure()

    labels = np.array(labels)

    # draw the projections in the figure
    for i, label_val in enumerate(np.unique(labels)):
        speaker_projs = X_r2[labels == label_val]
        plt.scatter(*speaker_projs.T, c=[_my_colors[i]], marker="+", label=label_val)
    
    # define the legend and the tile of the plot
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of ' + str(title))

    # save the LDA plot as an eps file
    plt.savefig(fig_name, format='eps', pad_inches=0)

def get_umap_lds_plots(codes, labels, label_types, f_str):
    """ Plot UMAP and LDA embeddings for each of the label types """
    
    for label_type in label_types:
        # define the umap plot file name
        fig_name = "umap-result_" + str(label_type) + "_" + f_str + ".eps"

        # plot the umap plot
        plot_umap(codes, labels[label_type], legend_title=label_type, fig_name=fig_name)
        
        if(not (("speaker_id" in label_type) and ("test" in f_str))):
            # define the LDA plot file name
            fig_name = "lda-result_" + str(label_type) + "_" + f_str + ".eps"
            
            # plot the LDA plot
            plot_lda(codes, labels[label_type], label_type, fig_name)

################ Code/Labels Manipulation Functions ################
def find_largest_code(codes):
    """ Find the largest code in the codes list and return its length """
    
    largest_code = 0
    for code in codes:
        if(code.shape[-1] > largest_code):
            largest_code = code.shape[-1]

    return largest_code

def pad_codes(codes):
    """
        Pad the codes in the codes list so that all the codes
        have the same length and return the padded codes
    """
    largest_code = find_largest_code(codes)

    padded_codes = []

    # for each code
    for code in codes:
        # if the code is smaller than the max code length
        if(code.shape[-1] < largest_code):
            len_pad = largest_code - code.shape[-1]
            assert len_pad >= 0

            # pad the code
            padded_code = np.pad(code, (0,len_pad), "constant", constant_values=(0,0))
            padded_codes.append(padded_code)
        else:
            padded_codes.append(code)
    
    # return the codes padded to the max code length
    return padded_codes

def arrange_labels(code, emotion_pred, utt_label):
    """ One entry for each utterance """
    preds = []
    if (len(emotion_pred) > 0):
        preds.append(emotion_pred)

    return [code.flatten()], [utt_label["emotion"]], [utt_label["speaker_id"]], preds

def get_session(subdir):
    """ Get the session number (1 to 5) """
    
    if("1" in subdir):
        return 1
    elif("2" in subdir):
        return 2
    elif("3" in subdir):
        return 3
    elif("4" in subdir):
        return 4
    elif("5" in subdir):
        return 5
    else:
        print("Error. Unexisting session number")
        sys.exit(1)

def get_gender(file_path, session):
    """ Get the gender as a string (Female/Male) and the speaker id as an integer """
    
    if("Female" in file_path):
        speaker_id = session + 5
        gender_str = "Female"
    elif("Male" in file_path):
        speaker_id = session
        gender_str = "Male"
    else:
        print("Error. Unexisting gender")
        sys.exit(1)
    return gender_str, speaker_id

def get_label_category(label_path, file_path):
    """ Return the emotion category, read from the csv file """

    # get the label file path
    file_name = file_path.split("/")[-1]
    file_name = file_name[:-4] + ".csv"
    file_path = label_path + "/" + file_name
    
    # open the label file
    with open(file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # return the emotion category as a string
            return row["category"]

def get_emo_label(session, gender_str, file_path, config):
    """ Return the emotion id for the data in file_path """

    label_main_path = config["label_path"]
    label_path = label_main_path + "/Session" + str(session) + "/" + str(gender_str)

    # get the emotion category as a string
    emotion_category = get_label_category(label_path=label_path, file_path=file_path)

    # get the emotion category as an int (according to the emotion_classes dict)
    if(emotion_category == "Excited"):
        emotion = emotion_classes["Happy"]
    else:
        emotion = emotion_classes[emotion_category]

    # return the emotion class as an int
    return emotion

def get_labels(file_path, config):
    """ Return the labels: emotion, speaker_id and content """

    file_name = file_path.split("/")[-1]
    subdir = file_name.split("_")[0]

    session = get_session(subdir)
    gender_str, speaker_id = get_gender(file_path, session)
    emotion = get_emo_label(session, gender_str, file_path, config)
    
    utt_labels = {}
    utt_labels["emotion"] = emotion
    utt_labels["speaker_id"] = speaker_id

    return utt_labels

################ Helpers for getting code and labels Functions ################
def pad_seq(x, base):
    """
        Pads the sequence x so that it has a length that is multiple
        of base. Used by the standard model.
    """
    len_out = int(base * ceil(float(x.shape[0])/base)) # the next multiple of base
    len_pad = len_out - x.shape[0] # get the pad length
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), "constant"), len_pad

def pad_onehot_seq(x, base):
    """ Pads the one hot sequence x. Used by the main model. """

    len_out = int(base * ceil(float(x.shape[0])/base)) # the next multiple of base
    len_pad = len_out - x.shape[0] # get the pad length
    assert len_pad >= 0
    padded_samples = np.pad(x, ((0,len_pad),(0,0)), "constant")
    padded_samples = padded_samples.tolist()

    num_pads = 0
    for sample in reversed(padded_samples):
        if(num_pads == len_pad):
            break
        else:
            sample[0] = 1
        num_pads += 1
    padded_samples = np.array(padded_samples)

    return padded_samples, len_pad

def get_wav2vec_tensor(file_name, wav2vec_path):
    """ Get wav2vec features from tensor """
    wav2vec_feat = np.load(os.path.join(wav2vec_path, file_name[:-4] + ".npy"))
    return wav2vec_feat

def get_model_input(config, x_org, sample, device):
    """ Obtain and zero pad the model input """

    content_emb_org=None
    wav2vec_feat = None

    pad_base = config["len_crop"]

    # get the phone sequence
    content_emb_org = sample[2]

    # get wav2vec feature
    wav2vec_feat = get_wav2vec_tensor(file_name=sample[0],
                                      wav2vec_path=config["wav2vec_path"])


    len_out = int(pad_base * ceil(float(wav2vec_feat.shape[1])/pad_base))
    len_pad = len_out - wav2vec_feat.shape[1] # get the pad length
    wav2vec_feat = np.pad(wav2vec_feat, ((0,0),(0,len_pad),(0,0)), "constant")

    # pad the spectrogram
    x_org, len_pad = pad_seq(x_org, pad_base)

    # pad the phone sequence (sequence of one-hot arrays)
    content_emb_org, len_pad_content = pad_onehot_seq(content_emb_org, pad_base)
    
    assert len_pad == len_pad_content
    
    content_emb_org = torch.from_numpy(content_emb_org[np.newaxis, :]).to(device, dtype=torch.float)

    return x_org, content_emb_org, wav2vec_feat

def init_labels_dict():
    """ Initialize the dictionary to store labels """
    
    keys = ["emotion", "speaker_id", "preds"]
    labels = {}
    for key in keys:
        labels[key] = []
    return labels

def get_code(sp_model, x_org, sample, content_emb_org, wav2vec_feat, device, config):
    """ Apply the input to the model and get the code """

    # transform the original speaker spectrogram and speaker embedding
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device) # original utterance
    emb_org = torch.from_numpy(sample[-2][np.newaxis, :]).to(device) # original speaker embedding
    wav2vec_feat = torch.from_numpy(wav2vec_feat[np.newaxis,:, :]).to(device)

    emotion_pred = []
    with torch.no_grad():
        codes = []
        emotion_preds = []
        num_frame_groups = int(uttr_org.shape[1]/config["len_crop"])
        for group_i in range(num_frame_groups):
            begin = int(group_i*config["len_crop"])
            end = int((group_i+1)*config["len_crop"])
            code, emotion_pred = sp_model(x=uttr_org[:,begin:end,:],
                                          spk_org=emb_org,
                                          spk_conv=None,
                                          cont_seq=content_emb_org[:,begin:end,:],
                                          wav2vec_feat=wav2vec_feat[:,:,begin:end,:])
            codes.append(code)
            emotion_preds.append(emotion_pred)
        code = torch.cat(codes, dim=-1)
        emotion_pred = torch.cat(emotion_preds, dim=-1)
        emotion_pred = emotion_pred.detach().cpu().numpy()
        emotion_pred = np.reshape(emotion_pred, (-1, 4))
    
    # reshape the code
    code = code.detach().cpu().numpy().flatten()
    
    # reshape according to the frame groups
    factor = int((config["len_crop"]/config["freq"])*(config["dim_neck"]*2))
    code = np.reshape(code, (int(code.shape[0]/factor), factor))
    
    return code, emotion_pred

def append_labels(codes_list, emotions, speaker_ids, preds, labels, codes):
    """ Append new labels to the labels dictionary """

    for i in codes_list:
        codes.append(i)
    for i in emotions:
        labels["emotion"].append(i)
    for i in speaker_ids:
        labels["speaker_id"].append(i)
    for i in preds:
        labels["preds"].append(i)
    return labels, codes

################ Get codes and Labels Functions ################
def get_codes_and_labels(data, config, sp_model, device, data_path):
    """ Get codes and labels (emotion, id, content, pred) information """
    
    codes = []
    labels = init_labels_dict()

    # for all utterances (data sample)
    for sample in data:
        # get the spectrogram file name
        spec_file_name = sample[0]
        
        # get the labels emotion, speaker_id
        utt_label = get_labels(file_path=spec_file_name,
                               config=config)

        # load the spectrogram data
        x_org = np.load(data_path + "/" + str(spec_file_name))
        
        # get the padded model inputs
        x_org, content_emb_org, wav2vec_feat = get_model_input(config=config,
                                                                x_org=x_org,
                                                                sample=sample,
                                                                device=device)
        # get the codes
        code, emotion_pred = get_code(sp_model, x_org, sample, content_emb_org, wav2vec_feat, device, config)

        codes_list, emotions, speaker_ids, preds = arrange_labels(code=code,
                                                                  emotion_pred=emotion_pred,
                                                                  utt_label=utt_label)

        labels, codes = append_labels(codes_list, emotions, speaker_ids, preds, labels, codes)

    return codes, labels

################ Functions to get data ################
def get_data(data_path):
    print("[INFO] Getting the data...")
    data = pickle.load(open(data_path, "rb"))

    return data

################ "Main" Function ################
def get_plots(sp_model, config, data_path, pkl_file, device, fold, time):

    f_str = time + "_fold" + str(fold)

    data = get_data(data_path + "/" + pkl_file + ".pkl")

    codes, labels = get_codes_and_labels(data=data,
                                         config=config,
                                         sp_model=sp_model,
                                         device=device,
                                         data_path=data_path)
    padded_codes = pad_codes(codes)
    get_umap_lds_plots(codes=padded_codes,
                        labels=labels,
                        label_types=["speaker_id", "emotion"],
                        f_str=f_str)
