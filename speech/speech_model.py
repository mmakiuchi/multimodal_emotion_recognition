"""
    Name: speech_model.py
    Description: Defines the speech model class.
"""

import torch
import torch.nn as nn

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import speech.models as models # models and custom layers

########### Helper functions ###########
def get_codes_len(config):
    """ Return the code's len (to define the classifier's input size) """
    codes_len = config.dim_neck*2 # the codes have dim_neck*2 columns
    num_frames = config.len_crop
    # the codes have num_frames/freq rows
    codes_len = codes_len*(int(num_frames/config.freq))
    return codes_len

########### Functions to get the submodels ###########
def get_classifier(config):
    """ Return the classifier """
    
    # get the codes len
    input_len = get_codes_len(config)

    # get the number of output classes
    num_classes = config.num_emo_classes

    # activation function
    use_sigmoid = False

    # get the classifier model
    classifier = models.MLP(input_len=input_len,
                            num_classes=num_classes,
                            use_drop=config.use_drop,
                            use_sigmoid=use_sigmoid)
    
    return classifier

def get_encoder(config):
    """ Return the encoder. """
    return models.Encoder(config)

def get_decoder(config):
    """ Return the decoder. """
    return models.Decoder(config)

def get_phone_encoder(config):
    """ Return the phone encoder model. """
    return models.PhoneEncoder(config=config)

class SpeechModel(nn.Module):
    """SpeechModel network."""
    def __init__(self, config):
        super(SpeechModel, self).__init__()

        # encoder
        self.encoder = get_encoder(config)
        
        # phone encoder
        self.phone_encoder = get_phone_encoder(config)

        # decoder
        self.decoder = get_decoder(config)

        # get emo classifier
        self.emo_classifier = get_classifier(config)
        
        # get the postnet
        self.postnet = models.Postnet(config=config)
        
    def forward(self, x, spk_org, spk_conv, cont_seq, wav2vec_feat):

        ###########################################################
        #               Get embeddings (i.e. codes)               #
        ###########################################################
        codes = self.encoder(x, spk_org, wav2vec_feat)

        # initialize logits
        emotion_logit = None

        ###########################################################
        #               Get the classification logits             #
        ###########################################################
        class_input = torch.cat(codes, dim=-1)
        
        # get the classifier predictions
        emotion_logit = self.emo_classifier(class_input)

        # for inference
        if spk_conv is None:
            # return codes and logits
            return torch.cat(codes, dim=-1), emotion_logit # return the codes
        
        tmp = []
        ###########################################################
        #                       Upsampling                        #
        ###########################################################
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)

        ###########################################################
        #   Get the content embedding and prepare decoder input   #
        ###########################################################
        # input phone sequence to phone encoder
        cont_emb = self.phone_encoder(cont_seq)

        # concatenation between the upsamples encoder outputs, the speaker embedding and content
        decoder_input = torch.cat((code_exp, spk_conv.unsqueeze(1).expand(-1,x.size(1),-1), cont_emb), dim=-1)

        ###########################################################
        #           Get the output before the postnet             #
        ###########################################################
        spec_out = self.decoder(decoder_input)

        spec_out_postnet = self.postnet(spec_out.transpose(2,1))
        spec_out_postnet = spec_out + spec_out_postnet.transpose(2,1)

        return spec_out, spec_out_postnet, torch.cat(codes, dim=-1), emotion_logit