"""
    Name: cls_helpers.py
    
    Description: Scripts to train a classifier on the speech embeddings for
                 speaker recognition
"""

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

# sys.path.append("../")
import torch.nn as nn
import torch
import time, datetime
import numpy as np
from speech.speech_model import get_codes_len

########### Define classifier model ###########
class MLP(nn.Module):
    """ MLP classifier """
    def __init__(self, input_len, num_classes):
        super(MLP, self).__init__()

        self.hidden1 = nn.Linear(input_len, 2048)
        self.hidden2 = nn.Linear(2048,1024)
        self.hidden3 = nn.Linear(1024,1024)
        self.output = nn.Linear(1024,num_classes)

    def forward(self, x):
        
        x = nn.Softplus()(self.hidden1(x))
        x = nn.Softplus()(self.hidden2(x))
        x = nn.Softplus()(self.hidden3(x))

        result = self.output(x)
        
        return result

def get_cls(sp_config, device):
    """ Return MLP model for the speaker-identity task """
    
    input_len = get_codes_len(sp_config)

    # speaker identity model
    c_id_spk_dep = MLP(input_len=input_len,
                       num_classes=10)
    c_id_spk_dep = c_id_spk_dep.to(device)

    return c_id_spk_dep

def get_dict_rep():
    """ Initializes a dictionary to create a detail acc report """
    dict_rep = {}

    dict_rep["emo_class"] = {}
    dict_rep["spk_id"] = {}
    dict_rep["recording"] = {}
    dict_rep["session"] = {}
    dict_rep["spk_record"] = {}

    return dict_rep

def update_dict_rep(dict_rep, spk_id, recording, session):
    """ Adds one to each category of the dictionary """

    dicts = ["spk_id", "recording", "session"]
    keys = [spk_id, recording, session]

    for dict_name, key in zip(dicts, keys):
        if not (key in dict_rep[dict_name].keys()):
            dict_rep[dict_name][key] = 1
        else:
            dict_rep[dict_name][key] += 1
    
    # speaker classification at each recording
    if not (spk_id in dict_rep["spk_record"].keys()):
        dict_rep["spk_record"][spk_id] = {}
    
    if not (recording in dict_rep["spk_record"][spk_id].keys()):
        dict_rep["spk_record"][spk_id][recording] = 1
    else:
        dict_rep["spk_record"][spk_id][recording] += 1

    return dict_rep

def get_vars_from_file_name(file_name):
    """ Get variables (id, recording, etc) from speech file name """

    file_splits = file_name[0].split("/")
    session = file_splits[0]
    gender = file_splits[1]
    recording = file_splits[2][:-4]
    recording = recording.split("_")[0] + "_" + recording.split("_")[1]
    spk_id = session + "_" + gender

    return spk_id, recording, session

def eval(sp_model, sp_config, cls_model, test_db, device, task, f_name):
    """ Evaluate speaker identity classifier on the test partition """

    # get the dictionary reports
    cls_rep = get_dict_rep()
    total_rep = get_dict_rep()

    with torch.no_grad():
        # evaluation mode
        cls_model = cls_model.eval()

        # variables initialization
        correct = 0
        total_samples = 0

        # for each data sample (utterance) in the dataloader
        for sample in test_db:
            if sp_config["speech_input"] == "wav2vec":
                file_name, id_gt, uttrs, spk_org, content_embs, _, _, wav2vec_feats = sample
            else:
                file_name, id_gt, uttrs, spk_org, content_embs, _, _ = sample
                wav2vec_feats = [None] * len(uttrs)
            
            # get variables from speech file name
            spk_id, recording, session = get_vars_from_file_name(file_name)

            spk_org = spk_org.to(device)
            spk_org = spk_org.type(torch.cuda.FloatTensor)

            # map the labels to numpy
            id_gt = id_gt.numpy()[0]

            # initialize the utterance's segments prediction lists
            all_preds = []

            # for each segment
            for uttr, wav2vec_feat, content_emb in zip(uttrs, wav2vec_feats, content_embs):

                uttr = uttr.to(device)
                uttr = uttr.type(torch.cuda.FloatTensor)

                content_emb = content_emb.to(device)
                content_emb = content_emb.type(torch.cuda.FloatTensor)

                if sp_config["speech_input"] == "wav2vec":
                    wav2vec_feat = wav2vec_feat.to(device)
                    wav2vec_feat = wav2vec_feat.type(torch.cuda.FloatTensor)

                # get the codes for the segment
                codes, _ = sp_model(uttr, spk_org, None, content_emb, wav2vec_feat) # when the 3rd arg is None, the model retuns the codes

                preds = cls_model(codes)

                # segment-level emo prediction
                preds = preds.detach().cpu().numpy()
                preds = preds[0]

                # apply softmax
                preds = list(np.exp(preds)/sum(np.exp(preds)))
                
                if(len(all_preds) == 0):
                    # there are no preds for the utterance yet (we are at the first segment)
                    all_preds = preds
                else:
                    # we sum the posterior probabilities for each emotion class
                    all_preds = [x + y for x, y in zip(preds, all_preds)]

            # get the utterance-level predictions
            utt_pred = -1

            # utterance-level prediction
            if(len(all_preds) > 0):
                utt_pred = all_preds.index(max(all_preds))

            # check if the predictions are correct
            if(utt_pred == id_gt):
                correct += 1
                cls_rep = update_dict_rep(dict_rep=cls_rep,
                                          spk_id=spk_id,
                                          recording=recording,
                                          session=session)
            
            total_rep = update_dict_rep(dict_rep=total_rep,
                                      spk_id=spk_id,
                                      recording=recording,
                                      session=session)
            # update the number of predictions
            total_samples += 1

        with open(f_name, "a") as txt_f:
            for key in cls_rep.keys():
                if not (key == "spk_record"):
                    for sub_key in cls_rep[key].keys():
                        print(str(task) + " " + str(key) + " " + str(sub_key) + " (correct/total_in_specific_category): {:.2f}".format(cls_rep[key][sub_key]*100/total_rep[key][sub_key]))
                        txt_f.write(str(task) + " " + str(key) + " " + str(sub_key) + " (correct/total_in_specific_category): {:.2f}".format(cls_rep[key][sub_key]*100/total_rep[key][sub_key]) + "\n")
                else:
                    for sub_key in cls_rep[key].keys():
                        for sub_sub_key in cls_rep[key][sub_key].keys():
                            print(str(task) + " " + str(key) + " " + str(sub_key) + " (correct/total_in_specific_category): " + str(sub_sub_key) + ": {:.2f}".format(cls_rep[key][sub_key][sub_sub_key]*100/total_rep[key][sub_key][sub_sub_key]))
                            txt_f.write(str(task) + " " + str(key) + " " + str(sub_key) + " (correct/total_in_specific_category): " + str(sub_sub_key) + ": {:.2f}".format(cls_rep[key][sub_key][sub_sub_key]*100/total_rep[key][sub_key][sub_sub_key]) + "\n")

        # write/print evaluation results
        with open(f_name, "a") as txt_f:
            print("------ Task: " + str(task) + " Total samples: " + str(total_samples) + "\n")
            print("------ Accuracy (utterance-level): ", correct/total_samples)
            txt_f.write("-------- Task: " + str(task) + " Total samples: " + str(total_samples) + "\n")
            txt_f.write("-------- Accuracy (utterance-level): " + str(float(correct/total_samples)) + "\n")


########### Train/Evaluate classifiers ###########
def train_model(sp_model, sp_config, cls_model, train_db, test_db, task, f_name, device):
    """ Train speaker classification model """

    sp_model = sp_model.eval()
    cls_model = cls_model.train()
    
    epochs = 10
    
    log_step = 50
    count = 0
    optimizer = torch.optim.Adam(cls_model.parameters(), 0.0001)

    start_time = time.time()
    for epoch in range(epochs):
        print("------- Epoch: ", epoch)
        for sample in train_db:
            count += 1

            if sp_config["speech_input"] == "wav2vec":
                _, id_gt, uttrs, spk_org, content_embs, _, _, wav2vec_feats = sample
            else:
                _, id_gt, uttrs, spk_org, content_embs, _, _ = sample
                wav2vec_feats = [None] * len(uttrs)
            
            spk_org = spk_org.to(device)
            spk_org = spk_org.type(torch.cuda.FloatTensor)
            id_gt = id_gt.to(device)
            id_gt = id_gt.type(torch.cuda.LongTensor)

            # for each segment
            for uttr, wav2vec_feat, content_emb in zip(uttrs, wav2vec_feats, content_embs):
                # zero grad
                optimizer.zero_grad()

                uttr = uttr.to(device)
                uttr = uttr.type(torch.cuda.FloatTensor)

                content_emb = content_emb.to(device)
                content_emb = content_emb.type(torch.cuda.FloatTensor)

                if sp_config["speech_input"] == "wav2vec":
                    wav2vec_feat = wav2vec_feat.to(device)
                    wav2vec_feat = wav2vec_feat.type(torch.cuda.FloatTensor)

                # get the codes for the segment
                with torch.no_grad():
                    codes, _ = sp_model(uttr, spk_org, None, content_emb, wav2vec_feat) # when the 3rd arg is None, the model retuns the codes

                # train the classifiers on the codes
                logit = cls_model(codes)

                # get the losses
                gt = id_gt
                loss = nn.CrossEntropyLoss()(logit, gt)
                
                # backward
                loss.backward()
                optimizer.step()

            # Print out training information.
            if((count % log_step) == 0):
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Utterance [{}/{}] ".format(et, count, len(train_db)*epochs)
                log += str(task) + " loss {:.4f}".format(loss.item())
                print(log) 

    eval(sp_model=sp_model,
         sp_config=sp_config,
         cls_model=cls_model,
         test_db=test_db,
         device=device,
         task=task,
         f_name=f_name)

def train_cls(cls_model, sp_model, sp_config, dbs, fold, time, device):
    """ Train the cls model """

    _, _, train_db_cl, test_db_cl = dbs
    task = "dep_id"

    print("[INFO] Training classifier for ", task)
    train_model(sp_model=sp_model,
                sp_config=sp_config,
                cls_model=cls_model,
                train_db=train_db_cl,
                test_db=test_db_cl,
                task=task,
                f_name=time + "_fold_" + str(fold) + ".txt",
                device=device)