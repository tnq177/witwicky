from __future__ import print_function
from __future__ import division

import os
import nmt.all_constants as ac

"""You can add your own configuration function to this file and select
it using `--proto function_name`."""


def base_config():
    config = {}

    ### Locations of input/output files

    # The name of the model
    config['model_name']        = 'model_name'

    # Source and target languages
    # Input files should be named with these as extensions
    config['src_lang']          = 'src_lang'
    config['trg_lang']          = 'trg_lang'

    # Directory to read input files from
    config['data_dir']          = './nmt/data/{}'.format(config['model_name'])

    # Directory to save models and other outputs in
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])

    # Pathname of log file
    config['log_file']          = './nmt/DEBUG.log'

    ### Model options

    # Filter out sentences longer than this (minus one for bos/eos)
    config['max_train_length']  = 1000

    # Vocabulary sizes
    config['src_vocab_size']    = 0
    config['trg_vocab_size']    = 0
    config['joint_vocab_size']  = 0
    config['share_vocab']       = False

    # Normalize word embeddings (Nguyen and Chiang, 2018)
    config['fix_norm']          = False

    # Tie word embeddings
    config['tie_mode']          = ac.ALL_TIED

    # Whether to learn position encodings
    config['learned_pos']       = False
    # Position encoding size
    config['max_pos_length']    = 1024
    
    # Layer sizes
    config['embed_dim']         = 512
    config['ff_dim']            = 512 * 4
    config['num_enc_layers']    = 6
    config['num_enc_heads']     = 8
    config['num_dec_layers']    = 6
    config['num_dec_heads']     = 8

    # Whether residual connections should bypass layer normalization
    # if True, layer-norm->dropout->add
    # if False, dropout->add->layer-norm (as in original paper)
    config['norm_in']           = True

    ### Dropout/smoothing options

    config['dropout']           = 0.3
    config['word_dropout']      = 0.1
    config['label_smoothing']   = 0.1

    ### Training options

    config['batch_sort_src']    = True
    config['batch_size']        = 4096
    config['weight_init_type']  = ac.XAVIER_NORMAL
    config['normalize_loss']    = ac.LOSS_TOK

    # Hyperparameters for Adam optimizer
    config['beta1']             = 0.9
    config['beta2']             = 0.999
    config['epsilon']           = 1e-8

    # Learning rate
    config['warmup_steps']      = 24000
    config['warmup_style']      = ac.NO_WARMUP
    config['lr']                = 3e-4
    config['lr_decay']          = 0.8 # if this is set to > 0, we'll do annealing
    config['start_lr']          = 1e-8
    config['min_lr']            = 1e-5
    config['patience']          = 3

    # Gradient clipping
    config['grad_clip']         = 1.0 # if no clip, just set it to some big value like 1e9

    ### Validation/stopping options

    config['max_epochs']        = 100
    config['validate_freq']     = 1.0 # eval every [this many] epochs
    config['val_per_epoch']     = True # if this true, we eval after every [validate_freq] epochs, otherwise by num of batches
    config['val_by_bleu']       = True

    # Undo BPE segmentation when validating
    config['restore_segments']  = True

    # How many of the best models to save
    config['n_best']            = 1

    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    ### Decoding options

    config['beam_size']         = 4
    config['beam_alpha']        = 0.6

    return config


def sk2en():
    config = base_config()

    config['model_name']        = 'sk2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'sk'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/sk2en'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    return config


def en2vi():
    config = base_config()

    config['model_name']        = 'en2vi'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'vi'
    config['data_dir']          = './nmt/data/en2vi'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    return config


def ar2en():
    config = base_config()

    config['model_name']        = 'ar2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'ar'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/ar2en'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['tie_mode']          = ac.TRG_TIED
    config['share_vocab']       = False
    return config


def de2en():
    config = base_config()

    config['model_name']        = 'de2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'de'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/de2en'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    return config


def he2en():
    config = base_config()

    config['model_name']        = 'he2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'he'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/he2en'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['tie_mode']          = ac.TRG_TIED
    return config


def it2en():
    config = base_config()

    config['model_name']        = 'it2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'it'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/it2en'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    return config


def en2ja():
    config = base_config()

    config['model_name']        = 'en2ja'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'ja'
    config['data_dir']          = './nmt/data/en2ja'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['tie_mode']          = ac.TRG_TIED
    return config


def wmt14_en2de():
    config = base_config()

    config['model_name']        = 'wmt14_en2de'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'de'
    config['data_dir']          = './nmt/data/en2de'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    config['dropout']           = 0.1
    config['word_dropout']      = 0.0
    config['joint_vocab_size']  = 32768
    config['tie_mode']          = ac.ALL_TIED
    config['share_vocab']       = True
    return config


def en2de():
    config = base_config()

    config['model_name']        = 'en2de'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'de'
    config['data_dir']          = './nmt/data/en2de'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    return config


def ha2en():
    config = base_config()

    config['model_name']        = 'ha2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'ha'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/ha2en'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    return config


def tu2en():
    config = base_config()

    config['model_name']        = 'tu2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'tu'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/tu2en'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    return config


def uz2en():
    config = base_config()

    config['model_name']        = 'uz2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'uz'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/uz2en'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    return config


def hu2en():
    config = base_config()

    config['model_name']        = 'hu2en'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'hu'
    config['trg_lang']          = 'en'
    config['data_dir']          = './nmt/data/hu2en'
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')
    return config
