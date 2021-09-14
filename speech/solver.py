"""
    Name: solver.py
    Description: defines the class Solver, that is used to load
                 train and evaluate the SpeechModel
"""
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from speech.speech_model import SpeechModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import datetime
import numpy as np
import csv
from pathlib import Path

def get_checkpoint_path(checkpoint_path_str):
    """ Return the checkpoint path if it exists """
    checkpoint_path = Path(checkpoint_path_str)

    if not checkpoint_path.exists():
        return None
    else:
        return checkpoint_path_str

class Solver(object):

    def __init__(self, vcc_loader, test_loader, train_eval, train_batch1, config):
        """Initialize configurations."""

        # Data loaders
        self.vcc_loader = vcc_loader
        self.test_loader = test_loader
        self.train_eval = train_eval
        self.train_batch1 = train_batch1

        # All the configurations
        self.config = config

        # Model configurations
        self.dim_neck = config.dim_neck

        # Training configurations
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.out_dir = config.out_dir
        self.test_out_file = config.out_dir + "/test_results.txt"

        # Miscellaneous
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.config.device = self.device
        self.log_step = config.log_step
        self.checkpoint_step = config.checkpoint_step

        self.speech_input = config.speech_input

        # Accuracy variables
        self.test_emo_acc = 0.0
        self.train_emo_acc = 0.0

        # get the pretrained model weight's path
        self.checkpoint_path = get_checkpoint_path(config.pretrained_model)

        # Build the model
        self.build_model()
            
    def build_model(self):
        """Build the SpeechModel model."""
        self.model = SpeechModel(self.config)        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 0.0001)

        if not (self.checkpoint_path == None):
            with open(self.test_out_file, "a") as txt_f:
                txt_f.write("Loading the weights at " + self.checkpoint_path + "\n")
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def sample_to_device(self, samp_vars, mode):
        """ Map the sample variables to the correct type and device """

        spec_gt, spk_org, content_emb, emotion_gt = samp_vars
        
        # map the variables to cuda and the correct type
        spec_gt = spec_gt.to(self.device)
        spec_gt = spec_gt.type(torch.cuda.FloatTensor)
        spk_org = spk_org.to(self.device)
        spk_org = spk_org.type(torch.cuda.FloatTensor)
        content_emb = content_emb.to(self.device)
        content_emb = content_emb.type(torch.cuda.FloatTensor)
        if mode=="train":
            emotion_gt = emotion_gt.to(self.device)
            emotion_gt = emotion_gt.type(torch.cuda.LongTensor)
        elif mode == "test":
            emotion_gt = emotion_gt.numpy()[0]

        return spec_gt, spk_org, content_emb, emotion_gt

    def eval(self, partition, iteration):
        """ Evaluate the model assuming that the data is
            defined by an Utterance class dataloader. """
        
        with torch.no_grad():
            # get the data loader
            if(partition=="test"):
                test_loader = self.test_loader
            elif(partition=="train"):
                test_loader = self.train_batch1
            
            # evaluation mode
            self.model = self.model.eval()
            
            # variables initialization
            emo_correct = 0
            total_samples = 0

            # for each spectrogram cut
            for sample in test_loader:
                # get the data samples variables
                wav2vec_feat = None
                if self.speech_input=="wav2vec":
                    spec_gt, spk_org, content_emb, emo_gt, wav2vec_feat = sample
                    wav2vec_feat = wav2vec_feat.to(self.device)
                    wav2vec_feat = wav2vec_feat.type(torch.cuda.FloatTensor)
                elif self.speech_input=="spec":
                    spec_gt, spk_org, content_emb, emo_gt = sample
                
                # map the variables to cuda and the correct type
                t_vars = self.sample_to_device(samp_vars=[spec_gt, spk_org, content_emb, emo_gt],
                                               mode="test")
                spec_gt, spk_org, content_emb, emo_gt = t_vars

                # get the emo predictions
                _, emo_pred = self.model(spec_gt, spk_org, None, content_emb, wav2vec_feat) # when the 3rd arg is None, self.model retuns the codes
                
                # transform the emotion prediction
                emo_pred = emo_pred.detach().cpu().numpy()
                emo_pred = emo_pred[0]
                # apply softmax
                emo_pred = list(np.exp(emo_pred)/sum(np.exp(emo_pred)))
                emo_pred = emo_pred.index(max(emo_pred))

                # check if the predictions are correct
                if(emo_pred == emo_gt):
                    emo_correct += 1

                # update the number of predictions
                total_samples += 1

            # write/print the evaluation result
            with open(self.test_out_file, "a") as txt_f:
                print("[Iter " + iteration + "] " + partition + " emotion accuracy (segment-level): ", emo_correct/total_samples)
                txt_f.write("[Iter " + iteration + "] " + partition + " emotion accuracy (segment-level): " + str(float(emo_correct/total_samples)) + "\n")
            
            return emo_correct/total_samples


    def utt_eval(self, partition, iteration):
        """ Perform an utterance-level evaluation.
            It assumes that the data loaders are defined by the EvalUtterances class. """

        with torch.no_grad():
            # get the dataset
            if(partition=="test"):
                test_loader = self.test_loader
            elif(partition=="train"):
                test_loader = self.train_eval
            # evaluation mode
            self.model = self.model.eval()

            # variables initialization
            emo_correct = 0
            total_samples = 0

            # for each data sample (utterance) in the dataloader
            for sample in test_loader:

                # get the sample variables
                if self.speech_input=="wav2vec":
                    spec_gts, spk_org, content_embs, emo_gt, wav2vec_feats = sample
                elif self.speech_input=="spec":
                    spec_gts, spk_org, content_embs, emo_gt = sample
                    wav2vec_feats = [None] * len(spec_gts)

                # map the variables to cuda and the correct type
                spk_org = spk_org.to(self.device)
                spk_org = spk_org.type(torch.cuda.FloatTensor)

                # map the labels to numpy
                emo_gt = emo_gt.numpy()[0]
                
                # initialize the utterance's segments prediction lists
                emo_preds = []

                # for each segment (spec and content emb) of data
                for spec_gt, content_emb, wav2vec_feat in zip(spec_gts, content_embs, wav2vec_feats):
                    spec_gt = spec_gt.to(self.device)
                    spec_gt = spec_gt.type(torch.cuda.FloatTensor)
                    content_emb = content_emb.to(self.device)
                    content_emb = content_emb.type(torch.cuda.FloatTensor)
                    if self.speech_input=="wav2vec":
                        wav2vec_feat = wav2vec_feat.to(self.device)
                        wav2vec_feat = wav2vec_feat.type(torch.cuda.FloatTensor)

                    # get the emo predictions for the segment
                    _, emo_pred = self.model(spec_gt, spk_org, None, content_emb, wav2vec_feat) # when the 3rd arg is None, self.model retuns the codes

                    # segment-level emo prediction
                    emo_pred = emo_pred.detach().cpu().numpy()
                    emo_pred = emo_pred[0]

                    # apply softmax
                    emo_pred = list(np.exp(emo_pred)/sum(np.exp(emo_pred)))
                    
                    if(len(emo_preds) == 0):
                        # there are no emo_pred for the utterance yet (we are at the first segment)
                        emo_preds = emo_pred
                    else:
                        # we sum the posterior probabilities for each emotion class
                        emo_preds = [x + y for x, y in zip(emo_pred, emo_preds)]

                # get the utterance-level predictions
                utt_emo_pred = -1

                # utterance-level emotion prediction
                if(len(emo_preds) > 0):
                    utt_emo_pred = emo_preds.index(max(emo_preds))

                # check if the predictions are correct
                if(utt_emo_pred == emo_gt):
                    emo_correct += 1

                # update the number of predictions
                total_samples += 1
            
            # update the train/test acc for early stopping
            if(partition=="test"):
                self.test_emo_acc = emo_correct/total_samples
            elif(partition=="train"):
                self.train_emo_acc = emo_correct/total_samples

            # write/print evaluation results
            with open(self.test_out_file, "a") as txt_f:
                print("Total samples: " + str(total_samples) + "\n")
                print("[Iter " + iteration + "] " + partition + " emotion accuracy (utterance-level): ", emo_correct/total_samples)
                txt_f.write("Total samples: " + str(total_samples) + "\n")
                txt_f.write("[Iter " + iteration + "] " + partition + " emotion accuracy (utterance-level): " + str(float(emo_correct/total_samples)) + "\n")
            return emo_correct/total_samples

    #=====================================================================================================================================#
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        keys = ["G/loss_id", "G/loss_id_psnt", "G/loss_emotion"]

        # initialize a dictionary with losses and acc results.
        eval_dict = {}
        eval_dict["iteration"] = 0
        for key in keys:
            eval_dict[key] = -1

        eval_dict["G/loss_emotion"] = -1
        eval_dict["acc_emo_train_utt"] = -1
        eval_dict["acc_emo_test_utt"] = -1
        eval_dict["acc_emo_train_seg"] = -1

        print("losses: ", keys)

        # Start training.
        print("Start training...")
        start_time = time.time()
        print(self.model)

        eval_csv_file = str(self.out_dir) + "/eval_result.csv"

        # write the header
        with open(eval_csv_file, "w") as csv_f:
            w = csv.DictWriter(csv_f, eval_dict.keys())
            w.writeheader()

        data_iter = iter(data_loader)
        for i in range(self.num_iters):
            # get data sample
            wav2vec_feat = None
            try:
                if self.speech_input=="wav2vec":
                    spec_gt, spk_org, content_emb, emotion_gt, wav2vec_feat = next(data_iter)
                    wav2vec_feat = wav2vec_feat.to(self.device)
                    wav2vec_feat = wav2vec_feat.type(torch.cuda.FloatTensor)
                elif self.speech_input=="spec":
                    spec_gt, spk_org, content_emb, emotion_gt = next(data_iter)
            except:
                data_iter = iter(data_loader)
                if self.speech_input=="wav2vec":
                    spec_gt, spk_org, content_emb, emotion_gt, wav2vec_feat = next(data_iter)
                    wav2vec_feat = wav2vec_feat.to(self.device)
                    wav2vec_feat = wav2vec_feat.type(torch.cuda.FloatTensor)
                elif self.speech_input=="spec":
                    spec_gt, spk_org, content_emb, emotion_gt = next(data_iter)
            
            # map all the variables to cuda and to the correct types
            t_vars = self.sample_to_device(samp_vars=[spec_gt, spk_org, content_emb, emotion_gt],
                                           mode="train")
            spec_gt, spk_org, content_emb, emotion_gt = t_vars

            self.reset_grad()
            
            # training mode
            self.model = self.model.train()
            
            # forward pass
            spec_out, spec_out_psnt, _, emotion_logit = self.model(spec_gt, spk_org, spk_org, content_emb, wav2vec_feat)

            # Reconstruction losess
            g_loss_id = F.mse_loss(spec_gt, spec_out) # loss of the model before the postnet
            g_loss_id_psnt = F.mse_loss(spec_gt, spec_out_psnt) # postnet loss

            # Emotion losses
            emotion_loss = nn.CrossEntropyLoss()(emotion_logit, emotion_gt)

            # Sum the losses
            g_loss = g_loss_id + g_loss_id_psnt + emotion_loss

            # Backward and optimize.
            g_loss.backward()
            self.optimizer.step()
            
            # Logging.
            loss = {}
            loss["G/loss_id"] = g_loss_id.item() # reconstruction loss

            eval_dict["iteration"] = i+1
            eval_dict["G/loss_id"] = g_loss_id.item()

            loss["G/loss_emotion"] = emotion_loss.item() # emotion loss
            eval_dict["G/loss_emotion"] = emotion_loss.item()
            
            loss["G/loss_id_psnt"] = g_loss_id_psnt.item() # postnet loss
            eval_dict["G/loss_id_psnt"] = g_loss_id_psnt.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
            
            # Save checkpoints.
            appended_test_acc = True
            appended_train_acc = True
            if ((i+1) % self.checkpoint_step == 0):
                print("Saving checkpoint and evaluating")
                
                # utterance-base evaluation on the test partition
                emo_utt_acc = self.utt_eval(partition="test",iteration=str(i+1))

                eval_dict["acc_emo_test_utt"] = emo_utt_acc

                # evaluate on the train partition
                if ((i+1) % (self.checkpoint_step*3) == 0):
                    # utterance-based accuracy
                    emo_utt_acc = self.utt_eval(partition="train",iteration=str(i+1))
                    
                    # segment-based accuracy
                    emo_seg_acc = self.eval(partition="train", iteration=str(i+1))
                    
                    eval_dict["acc_emo_train_utt"] = emo_utt_acc
                    eval_dict["acc_emo_train_seg"] = emo_seg_acc
                else:
                    appended_train_acc = False

                # save the checkpoint
                torch.save({"epoch": (i+1),
                            "model": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "g_loss_id": g_loss_id.item(),
                            "g_loss_id_psnt": g_loss_id_psnt.item()}, str(self.out_dir) + "/checkpoint_step_" + str(i+1) + "_neckdim_" + str(self.dim_neck) + ".ckpt")
            else:
                appended_test_acc = False
                appended_train_acc = False

            # Save last checkpoint.
            if((i+1) == self.num_iters):
                print("Saving last checkpoint and evaluating")

                # evaluate on the test and train partitions
                emo_utt_acc_test = self.utt_eval(partition="test",iteration="last")
                emo_utt_acc_train = self.utt_eval(partition="train",iteration="last")
                
                eval_dict["acc_emo_test_utt"] = emo_utt_acc_test
                eval_dict["acc_emo_train_utt"] = emo_utt_acc_train
                
                appended_test_acc = True
                appended_train_acc = True

                # save the last checkpoint model
                torch.save({"epoch": (i+1),
                            "model": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "g_loss_id": g_loss_id.item(),
                            "g_loss_id_psnt": g_loss_id_psnt.item()}, str(self.out_dir) + "/last_checkpoint_neckdim_" + str(self.dim_neck) + ".ckpt")

            if not appended_test_acc:
                eval_dict["acc_emo_test_utt"] = -1
            
            if not appended_train_acc:
                eval_dict["acc_emo_train_utt"] = -1
        
            # Writing the eval_dict in a csv file to be read with pandas
            # write the header
            with open(eval_csv_file, "a") as csv_f:
                w = csv.DictWriter(csv_f, eval_dict.keys())
                w.writerow(eval_dict)
            # restart the dictionary
            for key in eval_dict:
                eval_dict[key] = -1
