"""
    Name: train_txt_model.py
    
    Description: Trains a classifier on text embeddings
                 extracted with the huggingface transformer API.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from txt_model import TxtModel
import glob
import argparse
import time, datetime
# import random

import train_helpers as helper

# text model's definition
MODEL_NAME = "bert-large-uncased"
FEAT_TYPE = "last_hidden_state"
BASE_DIR = "./text"

FEAT_DIR = BASE_DIR + "/" + FEAT_TYPE + "/" + MODEL_NAME

emo_categories = {"Angry":0,
                  "Neutral":1,
                  "Happy":2,
                  "Sad":3}

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print("Device: ", device)

# setting the seed
# manualSeed = 10
# np.random.seed(manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)

# torch.cuda.manual_seed(manualSeed)
# torch.cuda.manual_seed_all(manualSeed)
# torch.backends.cudnn.enabled = False 
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, label_dir, sessions, max_seq_len):
        """ Initialize and preprocess the Utterances dataset."""
        self.sessions = sessions
        dataset = []
        npy_files = []
        
        # get the list of all numpy files
        for session in sessions:
            npy_files += glob.glob(FEAT_DIR + "/Session" + str(session) + "/*/*.npy")

        print("Number of npy files: ", len(npy_files))

        for npy_file in npy_files:
            # load the numpy file
            semantic_feat = np.load(npy_file)
            semantic_feat = semantic_feat[0]

            semantic_feat = semantic_feat[1:-1] # ignore [CLS] and [SEP]
            
            # zero-pad the embedding
            semantic_feat = np.pad(semantic_feat, ((0,max_seq_len-semantic_feat.shape[0]),(0,0)), mode="constant")

            semantic_feat = torch.from_numpy(semantic_feat)

            emo_label, is_4classes = helper.get_emotion_label(FEAT_DIR, label_dir, npy_file, emo_categories)
            if is_4classes:
                dataset.append([semantic_feat, emo_label])
        
        self.dataset = list(dataset)
        self.num_tokens = len(self.dataset)
        
        print('num tokens: ', self.num_tokens) # number of utterances
        print('Finished loading the dataset...')
        
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens


# def _init_fn():
#     # Worker init function of data.DataLoader
#     np.random.seed(manualSeed)

def get_loaders(label_dir, sessions, batch_size, max_seq_len):
    """ Return the data loaders """
    print("Getting the train dataset...")
    train_dataset = Utterances(label_dir, sessions["train"], max_seq_len)

    print("Getting the test dataset...")
    test_dataset = Utterances(label_dir, sessions["test"], max_seq_len)
    
    # if using a seed, set "worker_init_fn=_init_fn" in the DataLoader

    # get train dataset
    trainloader = data.DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
    
    # get test dataset
    testloader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0,
                                  drop_last=False)

    # get train loader for evaluation
    trainloader1batch = data.DataLoader(dataset=train_dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=0,
                                        drop_last=True)
    return trainloader, testloader, trainloader1batch

def evaluate(model, testloader, test_out_file, mode, largest_test_acc):
    """ Evaluate the model on the test dataset. """
    new_largest = False
    num_classes = 4

    # class_recall stores the true positive and the positive values
    # for each emotion class
    class_recall = helper.get_class_recall_dict(num_classes=num_classes)

    with torch.no_grad():
        model = model.eval()
        emo_total = 0
        emo_correct = 0

        for sample in testloader:
            # get the embedding, the label
            bert_emb, emotion_gt = sample

            bert_emb = bert_emb.to(device)
            bert_emb = bert_emb.type(torch.cuda.FloatTensor)
            emotion_gt = emotion_gt.to(device)
            emotion_gt = emotion_gt.type(torch.cuda.LongTensor)

            emo_pred = model(bert_emb)

            emo_pred = emo_pred.detach().cpu().numpy()
            emo_pred = emo_pred[0]

            # apply softmax
            emo_pred = list(np.exp(emo_pred)/sum(np.exp(emo_pred)))
            emo_pred = emo_pred.index(max(emo_pred))
            if(emo_pred == emotion_gt):
                emo_correct += 1
                class_recall[str(emotion_gt.detach().cpu().numpy()[0])]["tp"] += 1

            class_recall[str(emotion_gt.detach().cpu().numpy()[0])]["p"] += 1
            emo_total += 1
        
        with open(test_out_file, "a") as txt_f:
            print(mode + " emotion accuracy: ", emo_correct/emo_total)
            txt_f.write(mode + " emotion accuracy: " + str(float(emo_correct/emo_total)) + "\n")
            recall_total = 0

            for class_i in range(num_classes):
                print(mode + " class " + str(class_i) + " recall: ", class_recall[str(class_i)]["tp"]/class_recall[str(class_i)]["p"])
                txt_f.write(mode + " class " + str(class_i) + " recall: " + str(class_recall[str(class_i)]["tp"]/class_recall[str(class_i)]["p"]) + "\n")
                recall_total += class_recall[str(class_i)]["tp"]/class_recall[str(class_i)]["p"]
            
            print(mode + " average recall: ", recall_total/num_classes)
            txt_f.write(mode + " average recall: " + str(recall_total/num_classes) + "\n")

        if((mode=="test") and (emo_correct/emo_total > largest_test_acc)):
            largest_test_acc = emo_correct/emo_total
            new_largest = True
    
    return largest_test_acc, new_largest

def train(model, optimizer, trainloader, testloader, trainloader1batch, num_iters, fold):
    """ Main loop to train the model. """
    
    keys = ["loss_emotion"]
    largest_test_acc = 0
    largest_iter = 0
    checkpoint_step = int(num_iters/10)
    log_step = int(num_iters/100)

    # output file and folder names
    now = datetime.datetime.now()
    time_f = str(now.strftime("%d-%m-%Y_%H-%M-%S"))
    out_dir = "./text/txt_model_out"
    test_out_file = out_dir + "/" + MODEL_NAME + "_test_results_" + time_f + "_fold_" + str(fold) + ".txt"

    # Start training
    print("Start training...")
    start_time = time.time()
    data_iter = iter(trainloader)
    print(model)
    for i in range(num_iters):
        # get data sample
        try:
            bert_emb, emotion_gt = next(data_iter)
        except:
            data_iter = iter(trainloader)
            bert_emb, emotion_gt = next(data_iter)
        
        # map tensors to type and device
        bert_emb = bert_emb.to(device)
        bert_emb = bert_emb.type(torch.cuda.FloatTensor)
        emotion_gt = emotion_gt.to(device)
        emotion_gt = emotion_gt.type(torch.cuda.LongTensor)

        # Zero the gradients
        optimizer.zero_grad()

        # Training mode
        model = model.train()
        
        # Forward pass
        emotion_logit = model(bert_emb)
        
        # Loss
        emotion_loss = nn.CrossEntropyLoss()(emotion_logit, emotion_gt)

        # bacward and optimize
        emotion_loss.backward()
        optimizer.step()
        
        # Logging.
        loss = {}
        loss["loss_emotion"] = emotion_loss.item()

        # Print out training information.
        if (i+1) % log_step == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, num_iters)
            for tag in keys:
                log += ", {}: {:.4f}".format(tag, loss[tag])
            print(log)
        
        if (i+1) % checkpoint_step == 0:
            print("Saving checkpoint and evaluating")
            _, _ = evaluate(model, trainloader1batch, test_out_file, "train", largest_test_acc)
            largest_test_acc, new_largest = evaluate(model, testloader, test_out_file, "test", largest_test_acc)
            if new_largest:
                largest_iter = i+1
            
            torch.save({"epoch": (i+1),
                        "model": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}, str(out_dir) + "/" + MODEL_NAME + "_checkpoint_step_" + str(i+1) + "_" + time_f + "_fold_" + str(fold) + ".ckpt")
        if((i+1) == num_iters):
            print("Saving last checkpoint and evaluating")
            _, _ = evaluate(model, trainloader1batch, test_out_file, "train", largest_test_acc)
            largest_test_acc, new_largest = evaluate(model, testloader, test_out_file, "test", largest_test_acc)
            if new_largest:
                largest_iter = i+1
            
            torch.save({"epoch": (i+1),
                        "model": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}, str(out_dir) + "/" + MODEL_NAME +"_last_checkpoint_" + time_f + "_fold_" + str(fold) + ".ckpt")

    print("Largest test acc=" + str(largest_test_acc) + " at iteration " + str(largest_iter))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir", type=str, help="directory to label files")
    config = parser.parse_args()
    label_dir = config.label_dir

    # maximum number of tokens in the sequence
    max_seq_len = 122

    # for all the folds
    for fold in range(1,6):
        sessions = {}
        sessions["train"] = list(range(1,6))
        sessions["train"].pop(fold-1)
        sessions["test"] = [fold]
        print("Fold: ", fold)
        print("Train sessions: ", sessions["train"])
        print("Test sessions: ", sessions["test"])

        ######## get the data loaders
        print("Getting the data loaders...")
        trainloader, testloader, trainloader1batch = get_loaders(label_dir=label_dir,
                                                                sessions=sessions,
                                                                batch_size=4,
                                                                max_seq_len=max_seq_len)
    
        ######## get the model
        print("Getting the model...")
        model = TxtModel()

        optimizer = torch.optim.Adam(model.parameters(), 0.0001)
        model.to(device)

        ######## train
        print("Training the model...")
        train(model=model,
              optimizer=optimizer,
              trainloader=trainloader,
              testloader=testloader,
              trainloader1batch=trainloader1batch,
              num_iters=412800,
              fold=fold)

if __name__ == '__main__':
    main()
