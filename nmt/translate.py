import os
import time

import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

import nmt.all_constants as ac
import nmt.utils as ut
from nmt.model import Model
from nmt.data_manager import DataManager
import nmt.configurations as configurations


class Translator(object):
    def __init__(self, args):
        super(Translator, self).__init__()
        self.config = getattr(configurations, args.proto)()
        self.logger = ut.get_logger(self.config['log_file'])

        self.input_file = args.input_file
        self.model_file = args.model_file

        if self.input_file is None or self.model_file is None or not os.path.exists(self.input_file) or not os.path.exists(self.model_file):
            raise ValueError('Input file or model file does not exist')

        self.data_manager = DataManager(self.config)
        self.translate()

    def ids_to_trans(self, trans_ids):
        words = []
        word_ids = []
        for idx, word_idx in enumerate(trans_ids):
            if word_idx == ac.EOS_ID:
                break
            words.append(self.data_manager.trg_ivocab[word_idx])
            word_ids.append(word_idx)

        return u' '.join(words), word_ids

    def get_trans(self, probs, scores, symbols):
        sorted_rows = numpy.argsort(scores)[::-1]
        best_trans = None
        best_tran_ids = None
        beam_trans = []
        for i, r in enumerate(sorted_rows):
            trans_ids = symbols[r]
            trans_out, trans_out_ids = self.ids_to_trans(trans_ids)
            beam_trans.append(u'{} {:.2f} {:.2f}'.format(trans_out, scores[r], probs[r]))
            if i == 0: # highest prob trans
                best_trans = trans_out
                best_tran_ids = trans_out_ids

        return best_trans, best_tran_ids, u'\n'.join(beam_trans)

    def plot_head_map(self, mma, target_labels, target_ids, source_labels, source_ids, filename):
        """https://github.com/EdinburghNLP/nematus/blob/master/utils/plot_heatmap.py
        Change the font in family param below. If the system font is not used, delete matplotlib
        font cache https://github.com/matplotlib/matplotlib/issues/3590
        """
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)

        # put the major ticks at the middle of each cell
        ax.set_xticks(numpy.arange(mma.shape[1]) + 0.5, minor=False)
        ax.set_yticks(numpy.arange(mma.shape[0]) + 0.5, minor=False)

        # without this I get some extra columns rows
        # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
        ax.set_xlim(0, int(mma.shape[1]))
        ax.set_ylim(0, int(mma.shape[0]))

        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        # source words -> column labels
        ax.set_xticklabels(source_labels, minor=False, family='Source Code Pro')
        for xtick, idx in zip(ax.get_xticklabels(), source_ids):
            if idx == ac.UNK_ID:
                xtick.set_color('b')
        # target words -> row labels
        ax.set_yticklabels(target_labels, minor=False, family='Source Code Pro')
        for ytick, idx in zip(ax.get_yticklabels(), target_ids):
            if idx == ac.UNK_ID:
                ytick.set_color('b')

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')

    def translate(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = Model(self.config).to(device)
        self.logger.info('Restore model from {}'.format(self.model_file))
        model.load_state_dict(torch.load(self.model_file))
        model.eval()

        best_trans_file = self.input_file + '.best_trans'
        beam_trans_file = self.input_file + '.beam_trans'
        open(best_trans_file, 'w').close()
        open(beam_trans_file, 'w').close()

        num_sents = 0
        with open(self.input_file, 'r') as f:
            for line in f:
                if line.strip():
                    num_sents += 1
        all_best_trans = [''] * num_sents
        all_beam_trans = [''] * num_sents

        with torch.no_grad():
            self.logger.info('Start translating {}'.format(self.input_file))
            start = time.time()
            count = 0
            for (src_toks, original_idxs) in self.data_manager.get_trans_input(self.input_file):
                src_toks_cuda = src_toks.to(device)
                rets = model.beam_decode(src_toks_cuda)

                for i, ret in enumerate(rets):
                    probs = ret['probs'].cpu().detach().numpy().reshape([-1])
                    scores = ret['scores'].cpu().detach().numpy().reshape([-1])
                    symbols = ret['symbols'].cpu().detach().numpy()

                    best_trans, best_trans_ids, beam_trans = self.get_trans(probs, scores, symbols)
                    all_best_trans[original_idxs[i]] = best_trans + '\n'
                    all_beam_trans[original_idxs[i]] = beam_trans + '\n\n'

                    count += 1
                    if count % 100 == 0:
                        self.logger.info('  Translating line {}, average {} seconds/sent'.format(count, (time.time() - start) / count))

        model.train()

        with open(best_trans_file, 'w') as ftrans, open(beam_trans_file, 'w') as btrans:
            ftrans.write(''.join(all_best_trans))
            btrans.write(''.join(all_beam_trans))

        self.logger.info('Done translating {}, it takes {} minutes'.format(self.input_file, float(time.time() - start) / 60.0))
