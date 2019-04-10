import time
from os.path import join
from os.path import exists
from itertools import islice
from collections import Counter
import numpy
import torch

import nmt.utils as ut
import nmt.all_constants as ac

numpy.random.seed(ac.SEED)


class DataManager(object):

    def __init__(self, config):
        super(DataManager, self).__init__()
        self.logger = ut.get_logger(config['log_file'])

        self.src_lang = config['src_lang']
        self.trg_lang = config['trg_lang']
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.beam_size = config['beam_size']
        self.one_embedding = config['tie_mode'] == ac.ALL_TIED
        self.share_vocab = config['share_vocab']
        self.word_dropout = config['word_dropout']
        self.batch_sort_src = config['batch_sort_src']
        self.max_train_length = config['max_train_length']

        self.vocab_sizes = {
            self.src_lang: config['src_vocab_size'],
            self.trg_lang: config['trg_vocab_size'],
            'joint': config['joint_vocab_size']
        }

        self.data_files = {
            ac.TRAINING: {
                self.src_lang: join(self.data_dir, 'train.{}'.format(self.src_lang)),
                self.trg_lang: join(self.data_dir, 'train.{}'.format(self.trg_lang))
            },
            ac.VALIDATING: {
                self.src_lang: join(self.data_dir, 'dev.{}'.format(self.src_lang)),
                self.trg_lang: join(self.data_dir, 'dev.{}'.format(self.trg_lang))
            },
            ac.TESTING: {
                self.src_lang: join(self.data_dir, 'test.{}'.format(self.src_lang)),
                self.trg_lang: join(self.data_dir, 'test.{}'.format(self.trg_lang))
            }
        }
        self.tok_count_files = {
            ac.TRAINING: join(self.data_dir, 'train.count'),
            ac.VALIDATING: join(self.data_dir, 'dev.count'),
            ac.TESTING: join(self.data_dir, 'test.count')
        }
        self.ids_files = {
            ac.TRAINING: join(self.data_dir, 'train.ids'),
            ac.VALIDATING: join(self.data_dir, 'dev.ids'),
            ac.TESTING: join(self.data_dir, 'test.ids')
        }
        self.vocab_files = {
            self.src_lang: join(self.data_dir, 'vocab-{}.{}'.format(self.vocab_sizes[self.src_lang], self.src_lang)),
            self.trg_lang: join(self.data_dir, 'vocab-{}.{}'.format(self.vocab_sizes[self.trg_lang], self.trg_lang))
        }

        self.setup()

    def setup(self):
        self.create_all_vocabs()
        # it's important to check if we use one vocab before converting data to token ids
        self.parallel_data_to_token_ids(mode=ac.TRAINING)
        self.parallel_data_to_token_ids(mode=ac.VALIDATING)

        if exists(self.data_files[ac.TESTING][self.src_lang]) and exists(self.data_files[ac.TESTING][self.trg_lang]):
            self.parallel_data_to_token_ids(mode=ac.TESTING)

    def create_all_vocabs(self):
        self.create_vocabs()
        self.create_joint_vocab()
        self.src_vocab, self.src_ivocab, self.src_w2f = self.init_vocab(self.src_lang)
        self.trg_vocab, self.trg_ivocab, self.trg_w2f = self.init_vocab(self.trg_lang)

    def create_vocabs(self):
        def _write_vocab_file(vocab_dict, max_vocab_size, vocab_file):
            vocab_list = ac._START_VOCAB + sorted(vocab_dict, key=vocab_dict.get, reverse=True)
            if max_vocab_size == 0:
                pass
            elif len(vocab_list) < max_vocab_size:
                msg = '    The actual vocab size {} is < than your required {}\n'.format(
                    len(vocab_list), max_vocab_size)
                msg += '    Due to shameful reason, this cannot be handled automatically\n'
                msg += '    Please change the vocab size for {} to {}'.format(vocab_file, len(vocab_list))
                self.logger.info(msg)
                raise ValueError(msg)
            else:
                vocab_list = vocab_list[:max_vocab_size]

            open(vocab_file, 'w').close()
            with open(vocab_file, 'w') as fout:
                for idx, w in enumerate(vocab_list):
                    tok_freq = vocab_dict.get(w, 0)
                    fout.write(u'{} {} {}\n'.format(w, idx, tok_freq))

        src_file = self.data_files[ac.TRAINING][self.src_lang]
        src_vocab_file = self.vocab_files[self.src_lang]
        max_src_vocab_size = self.vocab_sizes[self.src_lang]

        trg_file = self.data_files[ac.TRAINING][self.trg_lang]
        trg_vocab_file = self.vocab_files[self.trg_lang]
        max_trg_vocab_size = self.vocab_sizes[self.trg_lang]

        self.logger.info('Create vocabulary of size {}/{}, {}/{}'.format(self.src_lang, max_src_vocab_size, self.trg_lang, max_trg_vocab_size))
        if exists(src_vocab_file) and exists(trg_vocab_file):
            self.logger.info('    Vocab files exist at {}/{}'.format(src_vocab_file, trg_vocab_file))

        src_vocab = Counter()
        trg_vocab = Counter()
        with open(src_file, 'r') as src_f, open(trg_file, 'r') as trg_f:
            count = 0
            for src_line, trg_line in zip(src_f, trg_f):
                count += 1
                if count % 10000 == 0:
                    self.logger.info('    processing line {}'.format(count))

                src_line = src_line.strip().split()
                trg_line = trg_line.strip().split()
                if 0 < len(src_line) <= self.max_train_length and 0 < len(trg_line) <= self.max_train_length:
                    src_vocab.update(src_line)
                    trg_vocab.update(trg_line)

        _write_vocab_file(src_vocab, max_src_vocab_size, src_vocab_file)
        _write_vocab_file(trg_vocab, max_trg_vocab_size, trg_vocab_file)

    def create_joint_vocab(self):
        if not self.one_embedding:
            return

        self.logger.info('Using one embedding so use one joint vocab')

        joint_vocab_file = join(self.data_dir, 'joint_vocab.{}-{}'.format(self.src_lang, self.trg_lang))
        subjoint_src_vocab_file = join(self.data_dir, 'joint_vocab.{}'.format(self.src_lang))
        subjoint_trg_vocab_file = join(self.data_dir, 'joint_vocab.{}'.format(self.trg_lang))

        # If all vocab files exist, return
        if exists(joint_vocab_file) and exists(subjoint_src_vocab_file) and exists(subjoint_trg_vocab_file):
            # Make sure to replace the vocab_files with the subjoint_vocab_files
            self.vocab_files[self.src_lang] = subjoint_src_vocab_file
            self.vocab_files[self.trg_lang] = subjoint_trg_vocab_file
            msg = """Joint vocab files exists at
                        {}
                        {}
                        {}""".format(joint_vocab_file, subjoint_src_vocab_file, subjoint_trg_vocab_file)
            self.logger.info(msg)
            return

        # Else, first combine the two word2freq from src + trg
        _, _, src_word2freq = self.init_vocab(self.src_lang)
        _, _, trg_word2freq = self.init_vocab(self.trg_lang)
        for special_tok in ac._START_VOCAB:
            del src_word2freq[special_tok]
            del trg_word2freq[special_tok]

        joint_word2freq = dict(Counter(src_word2freq) + Counter(trg_word2freq))

        # Save the joint vocab files
        joint_vocab_list = ac._START_VOCAB + sorted(joint_word2freq, key=joint_word2freq.get, reverse=True)
        if self.vocab_sizes['joint'] != 0 and len(joint_vocab_list) > self.vocab_sizes['joint']:
            self.logger.info('Cut off joint vocab size from {} to {}'.format(len(joint_vocab_list), self.vocab_sizes['joint']))
            joint_vocab_list = joint_vocab_list[:self.vocab_sizes['joint']]

        open(joint_vocab_file, 'w').close()
        with open(joint_vocab_file, 'w') as fout:
            for idx, w in enumerate(joint_vocab_list):
                tok_freq = joint_word2freq.get(w, 0)
                fout.write(u'{} {} {}\n'.format(w, idx, tok_freq))

        joint_vocab, _, _ = self.init_vocab('', joint_vocab_file)
        self.logger.info('Joint vocab has {} keys'.format(len(joint_vocab)))
        self.logger.info('Joint vocab saved in {}'.format(joint_vocab_file))

        if self.share_vocab:
            for lang in [self.src_lang, self.trg_lang]:
                self.vocab_files[lang] = joint_vocab_file
                vocab_mask = numpy.ones([len(joint_vocab)], dtype=numpy.float32)
                vocab_mask_file = join(self.data_dir, 'joint_vocab_mask.{}.npy'.format(lang))
                numpy.save(vocab_mask_file, vocab_mask)

            self.logger.info('Use the same vocab for both')
            return

        # Now generate the separate subjoint vocab, i.e. words that appear in both corresponding language's training
        # data and joint vocab
        # Also generate vocab mask
        for (lang, org_vocab, vocab_file) in [(self.src_lang, src_word2freq, subjoint_src_vocab_file), (self.trg_lang, trg_word2freq, subjoint_trg_vocab_file)]:
            self.logger.info('Creating subjoint vocab file for {} at {}'.format(lang, vocab_file))
            new_vocab_list = ac._START_VOCAB[::] + [word for word in joint_vocab_list if word in org_vocab]

            open(vocab_file, 'w').close()
            with open(vocab_file, 'w') as fout:
                for word in new_vocab_list:
                    fout.write(u'{} {} {}\n'.format(word, joint_vocab[word], joint_word2freq.get(word, 0)))

            self.vocab_files[lang] = vocab_file
            vocab_mask = numpy.zeros([len(joint_vocab)], dtype=numpy.float32)
            for word in new_vocab_list:
                vocab_mask[joint_vocab[word]] = 1.0
            vocab_mask_file = join(self.data_dir, 'joint_vocab_mask.{}.npy'.format(lang))
            numpy.save(vocab_mask_file, vocab_mask)
            self.logger.info('Save joint vocab mask for {} to {}'.format(lang, vocab_mask_file))

    def init_vocab(self, lang=None, fromfile=None):
        if not lang and not fromfile:
            raise ValueError('Must provide either src/trg lang or fromfile')
        elif fromfile:
            vocab_file = fromfile
        else:
            vocab_file = self.vocab_files[lang]

        self.logger.info('Initialize {} vocab from {}'.format(lang, vocab_file))

        if not exists(vocab_file):
            raise ValueError('    Vocab file {} not found'.format(vocab_file))

        counter = 0 # fall back to old vocab format "word freq\n"
        vocab, ivocab, word2freq = {}, {}, {}
        with open(vocab_file, 'r') as f:
            for line in f:
                if line.strip():
                    wif = line.strip().split()
                    if len(wif) == 2:
                        word = wif[0]
                        freq = int(wif[1])
                        idx = counter
                        counter += 1
                    elif len(wif) == 3:
                        word = wif[0]
                        idx = int(wif[1])
                        freq = int(wif[2])
                    else:
                        raise ValueError('Something wrong with this vocab format')

                    vocab[word] = idx
                    ivocab[idx] = word
                    word2freq[word] = freq

        return vocab, ivocab, word2freq

    def parallel_data_to_token_ids(self, mode=ac.TRAINING):
        src_file = self.data_files[mode][self.src_lang]
        trg_file = self.data_files[mode][self.trg_lang]

        joint_file = self.ids_files[mode]
        joint_tok_count = self.tok_count_files[mode]

        msg = 'Parallel convert tokens from {} & {} to ids and save to {}'.format(src_file, trg_file, joint_file)
        msg += '\nAlso save the approx tok count to {}'.format(joint_tok_count)
        self.logger.info(msg)

        if exists(joint_file) and exists(joint_tok_count):
            self.logger.info('    Token-id-ed data exists at {}'.format(joint_file))
            return

        open(joint_file, 'w').close()
        open(joint_tok_count, 'w').close()
        num_lines = 0
        tok_count = 0
        with open(src_file, 'r') as src_f, \
                open(trg_file, 'r') as trg_f, \
                open(joint_file, 'w') as tokens_f:

            for src_line, trg_line in zip(src_f, trg_f):
                src_toks = src_line.strip().split()
                trg_toks = trg_line.strip().split()

                if 0 < len(src_toks) <= self.max_train_length and 0 < len(trg_toks) <= self.max_train_length:
                    num_lines += 1
                    if num_lines % 10000 == 0:
                        self.logger.info('    converting line {}'.format(num_lines))

                    src_ids = [self.src_vocab.get(w, ac.UNK_ID) for w in src_toks] + [ac.EOS_ID]
                    trg_ids = [ac.BOS_ID] + [self.trg_vocab.get(w, ac.UNK_ID) for w in trg_toks]
                    tok_count += max(len(src_ids), len(trg_ids)) + 1
                    src_ids = map(str, src_ids)
                    trg_ids = map(str, trg_ids)
                    data = u'{}|||{}\n'.format(u' '.join(src_ids), u' '.join(trg_ids))
                    tokens_f.write(data)

        with open(joint_tok_count, 'w') as f:
            f.write('{}\n'.format(str(tok_count)))

    def replace_with_unk(self, data):
        drop_mask = numpy.random.choice([True, False], data.shape, p=[self.word_dropout, 1.0 - self.word_dropout])
        drop_mask = numpy.logical_and(drop_mask, data != ac.PAD_ID)
        data[drop_mask] = ac.UNK_ID

    def _prepare_one_batch(self, b_src_input, b_src_seq_length, b_trg_input, b_trg_seq_length):
        batch_size = len(b_src_input)
        max_src_length = max(b_src_seq_length)
        max_trg_length = max(b_trg_seq_length)

        src_input_batch = numpy.zeros([batch_size, max_src_length], dtype=numpy.int32)
        trg_input_batch = numpy.zeros([batch_size, max_trg_length], dtype=numpy.int32)
        trg_target_batch = numpy.zeros([batch_size, max_trg_length], dtype=numpy.int32)

        for i in range(batch_size):
            src_input_batch[i] = b_src_input[i] + (max_src_length - b_src_seq_length[i]) * [ac.PAD_ID]
            trg_input_batch[i] = b_trg_input[i] + (max_trg_length - b_trg_seq_length[i]) * [ac.PAD_ID]
            trg_target_batch[i] = b_trg_input[i][1:] + [ac.EOS_ID] + (max_trg_length - b_trg_seq_length[i]) * [ac.PAD_ID]

        return src_input_batch, trg_input_batch, trg_target_batch

    def prepare_batches(self, src_inputs, src_seq_lengths, trg_inputs, trg_seq_lengths, batch_size, mode=ac.TRAINING):
        if not src_inputs.size:
            return [], [], []

        # Sorting by src lengths
        # https://www.aclweb.org/anthology/W17-3203
        sorted_idxs = numpy.argsort(src_seq_lengths if self.batch_sort_src else trg_seq_lengths)
        src_inputs = src_inputs[sorted_idxs]
        src_seq_lengths = src_seq_lengths[sorted_idxs]
        trg_inputs = trg_inputs[sorted_idxs]
        trg_seq_lengths = trg_seq_lengths[sorted_idxs]

        src_input_batches = []
        trg_input_batches = []
        trg_target_batches = []

        s_idx = 0
        while s_idx < len(src_inputs):
            e_idx = s_idx + 1
            max_src_in_batch = src_seq_lengths[s_idx]
            max_trg_in_batch = trg_seq_lengths[s_idx]
            while e_idx < len(src_inputs):
                max_src_in_batch = max(max_src_in_batch, src_seq_lengths[e_idx])
                max_trg_in_batch = max(max_trg_in_batch, trg_seq_lengths[e_idx])
                count = (e_idx - s_idx + 1) * max(max_src_in_batch, max_trg_in_batch)
                if count > batch_size:
                    break
                else:
                    e_idx += 1

            batch_data = self._prepare_one_batch(
                src_inputs[s_idx:e_idx],
                src_seq_lengths[s_idx:e_idx],
                trg_inputs[s_idx:e_idx],
                trg_seq_lengths[s_idx:e_idx])

            if mode == ac.TRAINING:
                self.replace_with_unk(batch_data[0]) # src
                self.replace_with_unk(batch_data[1]) # trg

            src_input_batches.append(batch_data[0])
            trg_input_batches.append(batch_data[1])
            trg_target_batches.append(batch_data[2])
            s_idx = e_idx

        return src_input_batches, trg_input_batches, trg_target_batches

    def process_n_batches(self, n_batches_string_list):
        src_inputs = []
        src_seq_lengths = []
        trg_inputs = []
        trg_seq_lengths = []

        num_samples = 0
        for line in n_batches_string_list:
            data = line.strip()
            if data:
                num_samples += 1
                data = data.split('|||')
                _src_input = data[0].strip().split()
                _trg_input = data[1].strip().split()
                _src_input = list(map(int, _src_input))
                _trg_input = list(map(int, _trg_input))

                _src_len = len(_src_input)
                _trg_len = len(_trg_input)

                src_inputs.append(_src_input)
                src_seq_lengths.append(_src_len)
                trg_inputs.append(_trg_input)
                trg_seq_lengths.append(_trg_len)

        # convert to numpy array for sorting & reindexing
        src_inputs = numpy.array(src_inputs)
        src_seq_lengths = numpy.array(src_seq_lengths)
        trg_inputs = numpy.array(trg_inputs)
        trg_seq_lengths = numpy.array(trg_seq_lengths)

        return src_inputs, src_seq_lengths, trg_inputs, trg_seq_lengths

    def get_batch(self, mode=ac.TRAINING, num_preload=1000, alternate_batch_size=None):
        ids_file = self.ids_files[mode]
        shuffle = mode == ac.TRAINING
        if shuffle:
            # First we shuffle training data
            start = time.time()
            ut.shuffle_file(ids_file)
            self.logger.info('Shuffling {} takes {} seconds'.format(ids_file, time.time() - start))

        with open(ids_file, 'r') as f:
            while True:
                next_n_lines = list(islice(f, num_preload))
                if not next_n_lines:
                    break

                src_inputs, src_seq_lengths, trg_inputs, trg_seq_lengths = self.process_n_batches(next_n_lines)
                batches = self.prepare_batches(src_inputs, src_seq_lengths, trg_inputs, trg_seq_lengths, self.batch_size if alternate_batch_size is None else alternate_batch_size, mode=mode)
                for batch_data in zip(*batches):
                    yield (torch.from_numpy(x).type(torch.long) for x in batch_data)

    def get_trans_input(self, input_file):
        data = []
        data_lengths = []
        with open(input_file, 'r') as f:
            for line in f:
                toks = line.strip().split()
                data.append([self.src_vocab.get(w, ac.UNK_ID) for w in toks] + [ac.EOS_ID])
                data_lengths.append(len(data[-1]))

        data_lengths = numpy.array(data_lengths)
        sorted_idxs = numpy.argsort(data_lengths)
        data_lengths = data_lengths[sorted_idxs]
        data = numpy.array(data)[sorted_idxs]

        batch_size = self.batch_size // self.beam_size
        s_idx = 0
        while s_idx < len(data):
            e_idx = s_idx + 1
            max_in_batch = data_lengths[s_idx]
            while e_idx < len(data):
                max_in_batch = max(max_in_batch, data_lengths[e_idx])
                count = (e_idx - s_idx + 1) * (2 * max_in_batch)
                if count > batch_size:
                    break
                else:
                    e_idx += 1

            max_in_batch = max(data_lengths[s_idx:e_idx])
            src_inputs = numpy.zeros((e_idx - s_idx, max_in_batch), dtype=numpy.int32)
            for i in range(s_idx, e_idx):
                src_inputs[i - s_idx] = list(data[i]) + (max_in_batch - data_lengths[i]) * [ac.PAD_ID]
            original_idxs = sorted_idxs[s_idx:e_idx]
            s_idx = e_idx

            yield torch.from_numpy(src_inputs).type(torch.long), original_idxs
