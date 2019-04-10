TRG_TIED = 1 # use 1 embed for both trg in/out embed
ALL_TIED = 2 # use 1 embed for all three

# Special vocabulary symbols - we always put them at the start.
_PAD = u'_PAD_'
_BOS = u'_BOS_'
_EOS = u'_EOS_'
_UNK = u'_UNK_'
_START_VOCAB = [_PAD, _BOS, _EOS, _UNK]

PAD_ID = 0
BOS_ID = 1
# It's crucial that EOS_ID != 0 (see beam_search decoder)
EOS_ID = 2
UNK_ID = 3

TRAINING = 'training'
VALIDATING = 'validating'
TESTING = 'testing'

XAVIER_UNIFORM = 0
XAVIER_NORMAL = 1

EMBED_NORMAL = 0
EMBED_UNIFORM = 1
EMBED_UNIFORM_SCALE = 2

LOSS_NONE = 0
LOSS_TOK = 1
LOSS_BATCH = 2

ORG_WARMUP = 0
FIXED_WARMUP = 1
NO_WARMUP = 2
UPFLAT_WARMUP = 3

SEED = 147
