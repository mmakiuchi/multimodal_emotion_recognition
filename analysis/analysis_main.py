"""
    Name: analysis_main.py
    
    Description: Scripts for result analysis
"""
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from multi_ua_wa_helpers import get_multimodal_results, get_sp_ua_wa
from plot_helpers import get_plots
from cls_helpers import get_cls, train_cls

from db_helpers import get_dbs
from config_helpers import get_config

import torch
from text.txt_model import TxtModel
from speech.speech_model import SpeechModel
import datetime

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print("[INFO] Device: ", device)

def get_trained_model(w_file, config, modality):
    """ Return a pretrained model (speech or text) with the loaded weights. """

    if modality == "speech":
        model = SpeechModel(config)
    elif modality == "text":
        model = TxtModel()
    
    model = model.to(device)

    print("[INFO] Loading the weights...")
    checkpoint = torch.load(w_file, map_location=device)
    model.load_state_dict(checkpoint["model"])

    return model

def main(analysis_options):

    # configuration of the speech model
    sp_config, data_dirs = get_config()

    # weights of the pretrained speech and text models
    sp_weight_fs = ["speech/results/fold1/last_checkpoint_neckdim_8.ckpt",
                    "speech/results/fold2/last_checkpoint_neckdim_8.ckpt",
                    "speech/results/fold3/last_checkpoint_neckdim_8.ckpt",
                    "speech/results/fold4/last_checkpoint_neckdim_8.ckpt",
                    "speechsresults/fold5/last_checkpoint_neckdim_8.ckpt"]

    txt_weight_fs = ["text/txt_model_out/bert-large-uncased_last_checkpoint_DD-MM-YYYY_HH-MM-SS_fold_1.ckpt",
                     "text/txt_model_out/bert-large-uncased_last_checkpoint_DD-MM-YYYY_HH-MM-SS_fold_2.ckpt",
                     "text/txt_model_out/bert-large-uncased_last_checkpoint_DD-MM-YYYY_HH-MM-SS_fold_3.ckpt",
                     "text/txt_model_out/bert-large-uncased_last_checkpoint_DD-MM-YYYY_HH-MM-SS_fold_4.ckpt",
                     "text/txt_model_out/bert-large-uncased_last_checkpoint_DD-MM-YYYY_HH-MM-SS_fold_5.ckpt"]

    # get the time string to identify the experiment
    now = datetime.datetime.now()
    time = str(now.strftime("%d-%m-%Y_%H-%M-%S"))

    # weights for the text and speech models to be used during fusion
    txt_w = 1.0
    sp_w = 0.6

    # fold number initialization
    fold = 1

    # for all the folds
    for sp_weight_f, txt_weight_f in zip(sp_weight_fs, txt_weight_fs):
        print("[INFO] Analysis for fold ", fold)

        print("[INFO] Getting the speech model")
        sp_model = get_trained_model(sp_weight_f, sp_config, "speech")

        # get dataloaders for the fold
        test_db = None
        dbs = None
        if (analysis_options["multimodal"] or analysis_options["ua_wa"] or analysis_options["train_cl"]):
            print("[INFO] Getting dataloaders")
            dbs = get_dbs(config=sp_config,
                          train_dir=data_dirs["data_dir"] + "/fold" + str(fold) + "/train",
                          test_dir=data_dirs["data_dir"] + "/fold" + str(fold) + "/test",
                          train_dir_cl=data_dirs["data_dir_cl"] + "/fold" + str(fold) + "/train",
                          test_dir_cl=data_dirs["data_dir_cl"] + "/fold" + str(fold) + "/test")
            _, test_db, _, _ = dbs

        # get the multimodal results and the UA, WA
        if analysis_options["multimodal"]:
            print("[INFO] Getting multimodal UA WA metrics")
            txt_model = get_trained_model(txt_weight_f, None, "text")
            get_multimodal_results(sp_model=sp_model,
                                   txt_model=txt_model,
                                   testloader=test_db,
                                   fold=fold,
                                   device=device,
                                   time=time,
                                   txt_w=txt_w,
                                   sp_w=sp_w)
        
        # get the UA, WA results for the speech model
        elif analysis_options["ua_wa"]:
            print("[INFO] Getting UA WA metrics")
            get_sp_ua_wa(sp_model=sp_model,
                         testloader=test_db,
                         fold=fold,
                         device=device,
                         time=time)

        # plot the embeddings
        if analysis_options["plot"]:
            print("[INFO] Plotting UMAP and LDA plots")
            get_plots(sp_model=sp_model,
                       config=sp_config,
                       data_path=data_dirs["data_dir"] + "/fold" + str(fold) + "/train",
                       pkl_file="train",
                       device=device,
                       fold=fold,
                       time=time + "_train_")
            get_plots(sp_model=sp_model,
                       config=sp_config,
                       data_path=data_dirs["data_dir"] + "/fold" + str(fold) + "/test",
                       pkl_file="test",
                       device=device,
                       fold=fold,
                       time=time + "_test_")

        # train classifiers for speaker recognition
        if analysis_options["train_cl"]:
            print("[INFO] Training classifiers for speaker recognition")
            cls_model = get_cls(sp_config=sp_config,
                                 device=device)
            train_cls(cls_model=cls_model,
                      sp_model=sp_model,
                      sp_config=sp_config,
                      dbs=dbs,
                      fold=fold,
                      time=time,
                      device=device)
        
        fold += 1
    print("[INFO] Finished running the analysis for all the folds")

if __name__ == '__main__':
    analysis_options = {}
    analysis_options["ua_wa"] = True # compute UA and WA
    analysis_options["plot"] = True # get the feature plots
    analysis_options["train_cl"] = True # train speaker identity classifiers
    analysis_options["multimodal"] = True # get the multimodal results
    main(analysis_options)
