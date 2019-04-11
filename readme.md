# Transformer in Pytorch
[Toan Q. Nguyen](http://tnq177.github.io), University of Notre Dame.  

An implementation of [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) with [PyTorch](https://pytorch.org). An [early version](https://github.com/tnq177/nmt_text_from_non_native_speaker) of this code was used for [Neural Machine Translation of Text from Non-Native Speakers
](https://arxiv.org/abs/1808.06267).  

This code implements Neural Machine Translation system called Transformer from the paper [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). It expects bitext data of format ``{train, dev, test}.{src_lang, trg_lang}``. During training, the model is validated and best checkpoint can be saved based on either dev BLEU score or (label smoothed) dev perplexity. When training is finished, the best checkpoint is reloaded and used for decoding on test file. We support batched beam search, but currently it's quite adhoc. We assume that during training, if a ``batch_size`` doesn't cause OOM, during beam search we can use a batch size of ``batch_size//beam_size`` (see ``get_trans_input`` function in ``data_manager.py``).  

My rule of thumb for data preprocessing is (learned from fairseq's code):  

* tokenize data
* length-limit around 80 tokens
* learn bpe from that 80-token-long data
* apply bpe
* use all of them (**don't limit training sentence length again**)

To train a new model:  
* Write a new configuration function in ``configurations.py``  
* Put preprocessed data in ``nmt/data/model_name`` or as configured in ``data_dir`` option in your configuration function  
* Run: ``python3 -m nmt --proto config_name``  

The best checkpoint is saved to ``nmt/saved_models/model_name/model_name-SCORE.pth``. To decode, run ``python3 -m nmt --proto config_name --model-file path_to_checkpoint --input-file path_to_file_to_decode``.  

Please note that the ``n_best`` option is incorrect if set to > 1 as I currently save the best checkpoint in the format ``model_name-SCORE.pth`` so if two checkpoints have the same score, there is overwritten. However, setting to 1 will always save the best checkpoint.  

This code has been tested with only Python3.6 and PyTorch 1.0
## Hyperparameters
Many of hyperparameters are pretty important, take a look at ``all_constants.py`` before read on:

* ``norm_in``: If it's False, it's the default Transformer's computational sequence, that is, we do dropout-->residual-add-->layernorm. If false, it's goes layernorm-->dropout-->residual-add. See below image for visualization. The latter is noted in [The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation](https://arxiv.org/pdf/1804.09849.pdf) for ensuring good model performance. ![alt text](./residual.jpeg "Residual")

* ``fix_norm``: implement the fixnorm in this [paper](https://aclweb.org/anthology/N18-1031) (though a tad different)

* warmup: There are ``ORG_WARMUP``, ``FIXED_WARMUP``, ``NO_WARMUP``, and ``UPFLAT_WARMUP``. 
    - The ``ORG_WARMUP`` follows the formula in [original paper](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).  
    - ``FIXED_WARMUP`` means going from ``start_lr`` to ``lr`` in some warmup steps then decays with inverse sqrt of update steps.  
    - ``NO_WARMUP`` means no warmup at all, and learning rate is decayed if dev performance is not improving. We can decide to decay either with dev BLEU or dev perp by changing the option ``val_by_bleu``. We can control the decay patience for ``NO_WARMUP`` with ``patience`` option too.  
    - ``UPFLAT_WARMUP`` means learning rate increases linearly like ``FIXED_WARMUP`` then stays there and decays only if dev performance is not improving like ``NO_WARMUP``.  

* ``tied_mode``:
    - ``TRG_TIED``: Tie target input and output embeddings
    - ``ALL_TIED``: Tie source embedding and target input, output embeddings
    - ``share_vocab``: For ``ALL_TIED``. If False, we only share the embeddings for common types between source and target vocabularies. This is done by simply masking out those not in target vocabulary in the output layer before softmax (set to -inf). If True, we don't do the masking.

* ``max_train_length``: The maximum length of training sentences. Any sentence pair of length less than this is discarded during training.

* ``vocab_size``: If set to 0, it means we don't do vocabulary cut-off.

* ``joint_vocab_size``: Similar vocab size but for the joint vocabulary when we use ``ALL_TIED``.

* ``word_drop``: This is [word dropout](https://www.aclweb.org/anthology/W16-2323) but instead of setting word embeds to zero, we replace dropped tokens with UNK. 

### Suggestion
#### General
For low-resource (<500k sentences), try ``word_drop = 0.1``, ``dropout=0.3``. For datasets of around 100k-500k sentences, I find BPE of 8k-12k is enough. With this small vocabulary size, just set ``vocab_size`` and ``joint_vocab_size`` to 0 which means we use all of them. If languages are similar, such as English and German, their subword vocabs are >90+% overlapped so set ``share_vocab`` to True.

I find gradient clipping at 1.0 helps stabilize training a bit and yields a small improvement in perplexity, so I always do that. Long sentences are important resource for Transformer since it doesn't seem to generalize well to longer sentences than those seen during training (see [Training Tips for the Transformer Model](https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf)). For this reason, I suggest to set ``max_train_length`` to high value such as 1000.  

I find ``fix_norm`` no longer helps with Transformer + BPE. However, it speeds up training a lot in early epochs and the final performance is either slightly better or the same so give it a try.


#### To Norm or Not To Norm
Section 3 of [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf) suggests residual connection should be left untouched for healthy back-propagation. I conjecture this is why doing dropout-->residual-add-->layernorm (left side of above figure, let's call it NormRes) is difficult to train without warmup. On the other hand, with layernorm-->dropout-->residual-add (right side of above figure, let's call it TrueRes), we actually don't need warmup at all. However, in order to achieve good performance, it is important to still decay the learning rate. We choose to decay if the performance on dev set is not improving over some ``patience`` previous validations. We often set ``patience`` to 3.

Below figure shows the dev BLEU curve for Hungarian-English (from LORELEI) with different model sizes and different warmup steps. Note that for all TrueRes models we don't do warmup. We can see that while the NormRes learns well with 4 layers, going to 6 layers causes a huge drop in performance. With TrueRes, we can also train a 6-layer model for this dataset which gives almost 2 BLEU gain.

![alt text](./hu2en_bleus_curve.png "hu2en")

The story is quite different with Arabic-English (and other TED datasets) though. For this dataset, both NormRes and TrueRes end up at about the same BLEU score. However, we can see that NormRes is very sensitive to the warmup step. Note that the TED talk datasets are fairly big (around 200k examples each) so I always use 6 layers.
![alt text](./ar2en_bleus_curve.png "ar2en")

#### How long should we train
It's common to train Transformer for about 100k iterations. This works out to be around 4-50 epochs for Arabic-English. However, we can see that from epoch 50th to 100th we can still get some good gain. Note that for all models here we use batch size of 4096 tokens instead of 25k tokens. My general rule of thumb is for dataset of around 50k-500k examples, we should train around 100 epochs. Coming from LSTM, Transformer is so fast training a bit longer still doesn't seem to take much time. See table below for some stats:  

|                                         | ar2en | de2en | he2en | it2en |
|-----------------------------------------|-------|-------|-------|-------|
| # examples                              | 212k  | 166k  | 210k  | 202k  |
| # target tokens                         | 5.1M  | 3.8M  | 5M    | 4.7M  |
| Training speed (# target tokens/second) | 10.2k | 9.2k  | 10.2k | 9.3k  |
| Total time for 100 epochs (hours)       | ~19   | ~16   | ~18   | ~20   |  


## Benchmarks
Below are some benchmarks and comparison between this code and some published numbers. For all ``this-code`` models, we don't use warmup, start out with learning rate 3x10<sup>-4</sup>, and decay with factor 0.8 if dev BLEU is not improving compared to previous ``patience=3`` validations.

### LORELEI benchmarks
All use 8k BPEs. Detokenized BLEU.

|                                                            | ha   | hu   | tu   | uz   |
|------------------------------------------------------------|------|------|------|------|
| [Nguyen and Chiang](https://aclweb.org/anthology/N18-1031), LSTM | 22.3 | 27.9 | 22.2 | 21   |
| this-code (4layers, 4heads)                                | 25.2 | 30.2 | 24.1 | 24.1 |
| this-code (6layers, 8heads)                                | 24.2 | 32   | 24.6 | 24.7 |
| this-code + fixnorm (6layers, 8heads)                      | 25.1 | 31.8 | 25.5 | 24.9 |

### IWSLT/KFTT benchmarks
Tokenized BLEU to be comparable to previous works.  

* En-Vi from [Effective Approaches to Attention-based Neural Machine Translation](https://nlp.stanford.edu/projects/nmt/), use 8k joint BPE (I also added word-based number).
* KFTT En2Ja from [Incorporating Discrete Translation Lexicons into Neural Machine Translation](https://aclweb.org/anthology/D16-1162), Word-based.
* Others from [When and Why are Pre-trained Word Embeddings Useful for Neural Machine Translation?
](https://github.com/neulab/word-embeddings-for-nmt), 12k joint BPE.

I'm pretty surprised we got much better BLEU than the multilingual baseline. Note that all of my baselines are bilingual only.

|                                                                                                      | en2vi              | ar2en | de2en | he2en | it2en | KFTT en2ja           |
|------------------------------------------------------------------------------------------------------|--------------------|-------|-------|-------|-------|----------------------|
| [Massively Multilingual NMT-baseline](https://arxiv.org/abs/1903.00089)                              | ---                | 27.84 | 30.5  | 34.37 | 33.64 | ---                  |
| [Massively Multilingual NMT-multilingual](https://arxiv.org/abs/1903.00089)                          | ---                | 28.32 | 32.97 | 33.18 | 35.14 | ---                  |
| [SwitchOut](https://arxiv.org/pdf/1808.07512.pdf), word-based, transformer                           | 29.09              | ---   | ---   | ---   | ---   | ---                  |
| [duyvuleo's transformer dynet](https://github.com/duyvuleo/Transformer-DyNet), transformer, ensemble | 29.71 (word-based) | ---   | ---   | ---   | ---   | 26.55 (BPE+ensemble) |
| [Nguyen and Chiang](https://aclweb.org/anthology/N18-1031), LSTM, word-based                         | 27.5               | ---   | ---   | ---   | ---   | 26.2                 |
| this-code (BPE)                                                                                      | 31.71              | 33.15 | 37.83 | 38.79 | 40.22 | ---                  |
| this-code, word-based                                                                                | 29.47 (4layers)    | ---   | ---   | ---   | ---   | 31.28 (6layers)      |

### References
Parts of code/scripts are borrowed/inspired from:  

* https://github.com/pytorch/fairseq
* https://github.com/tensorflow/tensor2tensor
* https://github.com/EdinburghNLP/nematus/
* https://github.com/mila-iqia/blocks
* https://github.com/moses-smt/mosesdecoder

