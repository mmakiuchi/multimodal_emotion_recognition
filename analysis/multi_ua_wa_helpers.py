"""
    Name: multi_ua_wa_helpers.py
    
    Description: Gets the multimodal results (text and speech) and the UA and WA metrics
"""

import torch
import numpy as np

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

def get_utt_score(sample, model, device):
    """ Get the speech model's score """

    emo_preds = [] # utterance-level score
    num_segments = 0 # number of segments in an utterance
    
    uttrs, emb_org, content_embs, wav2vec_feats = sample
    for uttr, content_emb, wav2vec_feat in zip(uttrs, content_embs, wav2vec_feats):
        uttr = uttr.to(device)
        uttr = uttr.type(torch.cuda.FloatTensor)
        content_emb = content_emb.to(device)
        content_emb = content_emb.type(torch.cuda.FloatTensor)
        wav2vec_feat = wav2vec_feat.to(device)
        wav2vec_feat = wav2vec_feat.type(torch.cuda.FloatTensor)

        # get the emo prediction for the segment
        _, emo_pred = model(uttr, emb_org, None, content_emb, wav2vec_feat) # when the 3rd arg is None, the model retuns the codes

        emo_pred = emo_pred.detach().cpu().numpy()
        emo_pred = emo_pred[0]

        # apply softmax
        emo_pred = list(np.exp(emo_pred)/sum(np.exp(emo_pred)))
        num_segments += 1

        if(len(emo_preds) == 0):
            # there are no emo_pred for the utterance yet (we are at the first segment)
            emo_preds = emo_pred
        else:
            # we sum the posterior probabilities for each emotion class
            emo_preds = [x + y for x, y in zip(emo_pred, emo_preds)]

    # divide each probability by the number of segments
    emo_preds = [x/num_segments for x in emo_preds]

    return emo_preds

def get_txt_score(sample, model):
    """ Get the text model's score """

    score = model(sample)

    score = score.detach().cpu().numpy()
    score = score[0]

    # apply softmax
    score = list(np.exp(score)/sum(np.exp(score)))

    return score

def write_modality_result(emo_correct, class_recall, emo_total, num_classes, out_file, modality):
    """ Write the results of single or multiple modalities
        in the screen (print) and in a text file. """
    
    with open(out_file, "a") as txt_f:
        print(modality + " emotion accuracy: ", emo_correct/emo_total)
        txt_f.write(modality + " emotion accuracy: " + str(float(emo_correct/emo_total)) + "\n")
        recall_total = 0

        for class_i in range(num_classes):
            print(modality + " class " + str(class_i) + " recall: ", class_recall[str(class_i)]["tp"]/class_recall[str(class_i)]["p"])
            txt_f.write(modality + " class " + str(class_i) + " recall: " + str(class_recall[str(class_i)]["tp"]/class_recall[str(class_i)]["p"]) + "\n")
            recall_total += class_recall[str(class_i)]["tp"]/class_recall[str(class_i)]["p"]
        
        print(modality + " average recall: ", recall_total/num_classes)
        txt_f.write(modality + " average recall: " + str(recall_total/num_classes) + "\n")
        print("---------------------------------------------------------------")
        txt_f.write("---------------------------------------------------------------\n")


def write_eval_results(corrects, class_recalls, emo_total, txt_total, num_classes, out_file):
    """ Write the evaluation results: train and test accuracy and average recall """

    [sp_correct, txt_correct, fus_correct] = corrects
    [sp_recall, txt_recall, fus_recall] = class_recalls

    # write results for unimodalities
    write_modality_result(sp_correct, sp_recall, emo_total, num_classes, out_file, "Speech")
    write_modality_result(txt_correct, txt_recall, txt_total, num_classes, out_file, "Text")

    # write the fusion result
    write_modality_result(fus_correct, fus_recall, emo_total, num_classes, out_file, "Fusion")


def get_multimodal_results(sp_model, txt_model, testloader, fold, device, time, txt_w, sp_w):
    """ Combines the results for text and speech models and shows the final UA and WA metrics """

    # there are 4 emotion classes
    num_classes = 4

    class_recall = get_class_recall_dict(num_classes=num_classes)
    sp_class_recall = get_class_recall_dict(num_classes=num_classes)
    txt_class_recall = get_class_recall_dict(num_classes=num_classes)

    # name of the output file with the ua and wa results
    out_file = time + "_multimodal_ua_wa_fold" + str(fold) + ".txt"

    # write the weights given to each modality
    with open(out_file, "w") as txt_f:
        print("[INFO] Text model weight: ", txt_w)
        txt_f.write("Text model weight: " + str(txt_w) + "\n")

        print("[INFO] Speech model weight: ", sp_w)
        txt_f.write("Speech model weight: " + str(sp_w) + "\n")

    with torch.no_grad():
        emo_total = 0
        emo_correct = 0
        sp_correct = 0
        txt_correct = 0
        txt_total = 0

        sp_model = sp_model.eval()
        txt_model = txt_model.eval()

        for sample in testloader:
            _, _, uttrs, emb_org, content_embs, emotion_gt, txt_emb, wav2vec_feats = sample

            # map the tensors to the correct device and data type
            txt_emb = txt_emb.to(device)
            txt_emb = txt_emb.type(torch.cuda.FloatTensor)
            emb_org = emb_org.to(device)
            emb_org = emb_org.type(torch.cuda.FloatTensor)
            emotion_gt = emotion_gt.to(device)
            emotion_gt = emotion_gt.type(torch.cuda.LongTensor)

            # speech sample to be inputted to the speech model
            sp_sample = [uttrs, emb_org, content_embs, wav2vec_feats]

            # get the fused score
            sp_score = get_utt_score(sp_sample, sp_model, device)

            if len(txt_emb.shape) == 2:
                # there is no text data for this sample
                score = sp_score
            else:
                # there is text data, so we merge the scores
                txt_score = get_txt_score(txt_emb, txt_model)
                score = [sp_w*x + txt_w*y for x, y in zip(sp_score, txt_score)]

            # get the unimodal prediction
            sp_pred = sp_score.index(max(sp_score))
            if len(txt_emb.shape) == 2:
                txt_pred = None
            else:
                txt_pred = txt_score.index(max(txt_score))

            # get the fused prediction
            emo_pred = score.index(max(score))

            # check whether the predictions are correct
            if(sp_pred == emotion_gt):
                sp_correct += 1
                sp_class_recall[str(emotion_gt.detach().cpu().numpy()[0])]["tp"] += 1
            if((not txt_pred==None) and (txt_pred == emotion_gt)):
                txt_correct += 1
                txt_class_recall[str(emotion_gt.detach().cpu().numpy()[0])]["tp"] += 1
            if(emo_pred == emotion_gt):
                emo_correct += 1
                class_recall[str(emotion_gt.detach().cpu().numpy()[0])]["tp"] += 1

            class_recall[str(emotion_gt.detach().cpu().numpy()[0])]["p"] += 1
            sp_class_recall[str(emotion_gt.detach().cpu().numpy()[0])]["p"] += 1
            if (not txt_pred == None):
                txt_class_recall[str(emotion_gt.detach().cpu().numpy()[0])]["p"] += 1
                txt_total += 1
            emo_total += 1

        write_eval_results(corrects=[sp_correct, txt_correct, emo_correct],
                           class_recalls=[sp_class_recall, txt_class_recall, class_recall],
                           emo_total=emo_total,
                           txt_total=txt_total,
                           num_classes=num_classes,
                           out_file=out_file)


def get_sp_ua_wa(sp_model, testloader, fold, device, time):
    """ Computes and prints the UA, WA metrics for the speech model """

    # 4 emotion classes
    num_classes = 4

    sp_class_recall = get_class_recall_dict(num_classes=num_classes)
    out_file = time + "_speech_ua_wa_fold" + str(fold) + ".txt"

    with torch.no_grad():
        emo_total = 0
        sp_correct = 0
        sp_model = sp_model.eval()

        for sample in testloader:
            
            _, _, uttr, emb_org, content_emb, emotion_gt, txt_emb, wav2vec_feats = sample

            txt_emb = txt_emb.to(device)
            txt_emb = txt_emb.type(torch.cuda.FloatTensor)
            emb_org = emb_org.to(device)
            emb_org = emb_org.type(torch.cuda.FloatTensor)
            emotion_gt = emotion_gt.to(device)
            emotion_gt = emotion_gt.type(torch.cuda.LongTensor)
            
            sp_sample = [uttr, emb_org, content_emb, wav2vec_feats]

            # get the fused score
            sp_score = get_utt_score(sp_sample, sp_model, device)

            # get the unimodal prediction
            sp_pred = sp_score.index(max(sp_score))

            # check whether the predictions are correct
            if(sp_pred == emotion_gt):
                sp_correct += 1
                sp_class_recall[str(emotion_gt.detach().cpu().numpy()[0])]["tp"] += 1

            sp_class_recall[str(emotion_gt.detach().cpu().numpy()[0])]["p"] += 1
            emo_total += 1
        
        write_modality_result(emo_correct=sp_correct,
                              class_recall=sp_class_recall,
                              emo_total=emo_total,
                              num_classes=num_classes,
                              out_file=out_file,
                              modality="Speech")