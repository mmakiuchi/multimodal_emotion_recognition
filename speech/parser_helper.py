"""
    Name: parser_helper.py
    Description: helper functions to parse the configuration variables to run
                 the speech emotion recognition scripts
"""

import argparse

def argparser():
    """ Parse arguments """
    parser = argparse.ArgumentParser()
    
    # Directories
    parser.add_argument("--out_dir", type=str, default="speech/results", help="Folder to keep the models")
    parser.add_argument("--train_dir", type=str, default="speech/folds/fold5/train")
    parser.add_argument("--test_dir", type=str, default="speech/folds/fold5/test")
    parser.add_argument("--wav2vec_path", type=str, default="wav2vec/features/large-lv60/all_hidden_states", help="Path to all wav2vec features")

    # Reproduce experiments (small|large|spec) or run with free parameters (free)
    parser.add_argument("--reproduct_mode", type=str, default="free", help="small|large|spec modes to reproduce experiments or free mode")

    # Model configuration
    parser.add_argument("--dim_emb", type=int, default=256, help="dimension of the speaker id embedding")
    parser.add_argument("--dim_phone", type=int, default=128, help="how long is a phone") # 123 for one hot-encoding
    parser.add_argument("--dim_phone_emb", type=int, default=128, help="dimension of the phone embedding")
    parser.add_argument("--dim_pre", type=int, default=512)
    parser.add_argument("--use_drop", type=bool, default=True, help="if we apply dropout to the last layer of the classifier")
    parser.add_argument("--num_emo_classes", type=int, default=4, help="number of emotion classes")

    # Bottleneck configurations
    parser.add_argument("--dim_neck", type=int, default=32, help="bottleneck dimension") # code dimension (I think)
    parser.add_argument("--freq", type=int, default=16, help="frequency reduction on the time dimension at the bottleneck")

    # Input spectrogram configuration
    parser.add_argument("--len_crop", type=int, default=96, help="dataloader output sequence length")
    parser.add_argument("--num_mels", type=int, default=80, help="number of mel features at each frame (it should include delta and delta-delta if using those features)")
    parser.add_argument("--wav2vec_feat_len", type=int, default=1024, help="size of wav2vec features")
    parser.add_argument("--speech_input", type=str, default="wav2vec", help="type of speech representation to be input to the encoder")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=2, help="mini-batch size")
    parser.add_argument("--num_iters", type=int, default=500000, help="number of total iterations")
    parser.add_argument("--pretrained_model", type=str, default="none", help="path to the weights of a pretrained model")

    # Miscellaneous
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--checkpoint_step", type=int, default=50000, help="number of iterations to save model checkpoints")

    config = parser.parse_args()

    return config

def reproduct_config(config):
    """ Return the configuration to reproduce the experiments """

    config.dim_emb = 256
    config.dim_phone = 128
    config.dim_phone_emb = 128
    config.dim_pre = 512
    config.use_drop = True
    config.num_emo_classes = 4
    
    config.len_crop = 96
    config.num_mels = 80
    config.wav2vec_feat_len = 1024

    config.batch_size = 2
    config.num_iters = 1000000
    config.checkpoint_step = 1000001 # do not save checkpoints

    ## save checkpoints every 50k iterations
    # config.num_iters = 500000
    # config.checkpoint_step = 50000
    
    config.speech_input = "wav2vec"

    if config.reproduct_mode == "small":
        config.dim_neck = 8
        config.freq = 48
    elif config.reproduct_mode == "large":
        config.dim_neck = 128
        config.freq = 2
    elif config.reproduct_mode == "spec":
        config.speech_input = "spec"
        config.dim_neck = 8
        config.freq = 48

    return config

def get_config():
    """ Return the configuration to run the scripts """
    config = argparser()

    # use the configurations to reproduce the experiments
    if not config.reproduct_mode == "free":
        config = reproduct_config(config)
    
    print(config)

    with open(config.out_dir + "/test_results.txt", "w") as txt_f:
        txt_f.write(str(config) + "\n")

    return config
