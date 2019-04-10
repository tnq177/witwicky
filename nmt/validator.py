import os
import re
import time
import shutil
from subprocess import Popen, PIPE
from os.path import join
from os.path import exists

import numpy
import torch

import nmt.utils as ut
import nmt.all_constants as ac


class Validator(object):
    def __init__(self, config, data_manager):
        super(Validator, self).__init__()
        self.logger = ut.get_logger(config['log_file'])
        self.logger.info('Initializing validator')

        self.data_manager = data_manager
        self.restore_segments = config['restore_segments']
        self.beam_alpha = config['beam_alpha']
        self.val_by_bleu = config['val_by_bleu']

        self.get_cpkt_path = lambda score: join(config['save_to'], '{}-{}.pth'.format(config['model_name'], score))
        self.n_best = config['n_best']

        self.bleu_script = './scripts/multi-bleu.perl'
        assert exists(self.bleu_script)

        self.save_to = config['save_to']
        if not exists(self.save_to):
            os.makedirs(self.save_to)

        self.val_trans_out = config['val_trans_out']
        self.val_beam_out = config['val_beam_out']

        # I'll leave test alone for now since this version of the code doesn't automatically
        # report BLEU on test anw. The reason is it's up to the dataset to use multi-bleu
        # or NIST bleu. I'll include it in the future
        self.dev_ref = self.data_manager.data_files[ac.VALIDATING][self.data_manager.trg_lang]
        if self.restore_segments:
            self.dev_ref = self.remove_bpe(self.dev_ref)

        self.perp_curve_path = join(self.save_to, 'dev_perps.npy')
        self.best_perps_path = join(self.save_to, 'best_perp_scores.npy')
        self.perp_curve = numpy.array([], dtype=numpy.float32)
        self.best_perps = numpy.array([], dtype=numpy.float32)
        if exists(self.perp_curve_path):
            self.perp_curve = numpy.load(self.perp_curve_path)
        if exists(self.best_perps_path):
            self.best_perps = numpy.load(self.best_perps_path)

        if self.val_by_bleu:
            self.bleu_curve_path = join(self.save_to, 'bleu_scores.npy')
            self.best_bleus_path = join(self.save_to, 'best_bleu_scores.npy')
            self.bleu_curve = numpy.array([], dtype=numpy.float32)
            self.best_bleus = numpy.array([], dtype=numpy.float32)
            if exists(self.bleu_curve_path):
                self.bleu_curve = numpy.load(self.bleu_curve_path)
            if exists(self.best_bleus_path):
                self.best_bleus = numpy.load(self.best_bleus_path)

    def _ids_to_trans(self, trans_ids):
        words = []
        for idx in trans_ids:
            if idx == ac.EOS_ID:
                break
            words.append(self.data_manager.trg_ivocab[idx])

        return u' '.join(words)

    def get_trans(self, probs, scores, symbols):
        sorted_rows = numpy.argsort(scores)[::-1]
        best_trans = None
        beam_trans = []
        for i, r in enumerate(sorted_rows):
            trans_ids = symbols[r]
            trans_out = self._ids_to_trans(trans_ids)
            beam_trans.append(u'{} {:.2f} {:.2f}'.format(trans_out, scores[r], probs[r]))
            if i == 0: # highest prob trans
                best_trans = trans_out

        return best_trans, u'\n'.join(beam_trans)

    def evaluate_perp(self, model):
        model.eval()
        start_time = time.time()
        total_loss = 0.
        total_smoothed_loss = 0.
        total_weight = 0.

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for batch_data in self.data_manager.get_batch(mode=ac.VALIDATING):
                src_toks, trg_toks, targets = batch_data

                src_toks_cuda = src_toks.to(device)
                trg_toks_cuda = trg_toks.to(device)
                targets_cuda = targets.to(device)

                # get loss
                ret = model(src_toks_cuda, trg_toks_cuda, targets_cuda)
                total_loss += ret['nll_loss'].cpu().detach().numpy()
                total_smoothed_loss += ret['loss'].cpu().detach().numpy()
                total_weight += (targets != ac.PAD_ID).detach().numpy().sum()

        perp = total_loss / total_weight
        perp = numpy.exp(perp) if perp < 300 else float('inf')
        perp = round(perp, ndigits=3)

        smoothed_perp = total_smoothed_loss / total_weight
        smoothed_perp = numpy.exp(smoothed_perp) if smoothed_perp < 300 else float('inf')
        smoothed_perp = round(smoothed_perp, ndigits=3)

        self.perp_curve = numpy.append(self.perp_curve, smoothed_perp)
        numpy.save(self.perp_curve_path, self.perp_curve)

        model.train()
        self.logger.info('dev perplexity: {}'.format(perp))
        self.logger.info('smoothed dev perplexity: {}'.format(smoothed_perp))
        self.logger.info('Calculating dev perp took: {} minutes'.format(float(time.time() - start_time) / 60.0))

    def evaluate_bleu(self, model):
        model.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        val_trans_out = self.val_trans_out
        val_beam_out = self.val_beam_out
        ref_file = self.dev_ref

        if self.restore_segments: # using bpe
            val_trans_out = val_trans_out + '.bpe'
            val_beam_out = val_beam_out + '.bpe'

        src_file = self.data_manager.data_files[ac.VALIDATING][self.data_manager.src_lang]
        num_sents = 0
        with open(src_file, 'r') as f:
            for line in f:
                if line.strip():
                    num_sents += 1
        all_best_trans = [''] * num_sents
        all_beam_trans = [''] * num_sents

        start = time.time()
        with torch.no_grad():
            start_time = time.time()
            count = 0
            for (src_toks, original_idxs) in self.data_manager.get_trans_input(src_file):
                src_toks_cuda = src_toks.to(device)
                rets = model.beam_decode(src_toks_cuda)

                for i, ret in enumerate(rets):
                    probs = ret['probs'].cpu().detach().numpy().reshape([-1])
                    scores = ret['scores'].cpu().detach().numpy().reshape([-1])
                    symbols = ret['symbols'].cpu().detach().numpy()

                    best_trans, beam_trans = self.get_trans(probs, scores, symbols)
                    all_best_trans[original_idxs[i]] = best_trans + '\n'
                    all_beam_trans[original_idxs[i]] = beam_trans + '\n\n'

                    count += 1
                    if count % 100 == 0:
                        self.logger.info('  Translating line {}, average {} seconds/sent'.format(count, (time.time() - start) / count))

        model.train()

        open(val_trans_out, 'w').close()
        open(val_beam_out, 'w').close()
        with open(val_trans_out, 'w') as ftrans, open(val_beam_out, 'w') as btrans:
            ftrans.write(''.join(all_best_trans))
            btrans.write(''.join(all_beam_trans))

        # Remove BPE
        if self.restore_segments:
            val_trans_out = self.remove_bpe(val_trans_out, self.val_trans_out)
            val_beam_out = self.remove_bpe(val_beam_out, self.val_beam_out)

        multibleu_cmd = ['perl', self.bleu_script, ref_file, '<', val_trans_out]
        p = Popen(' '.join(multibleu_cmd), shell=True, stdout=PIPE)
        output, _ = p.communicate()
        out_parse = re.match(r'BLEU = [-.0-9]+', output.decode('utf-8'))
        self.logger.info(output)
        self.logger.info('Validation took: {} minutes'.format(float(time.time() - start_time) / 60.0))

        bleu = float('-inf')
        if out_parse is None:
            msg = '\n    Error extracting BLEU score, out_parse is None'
            msg += '\n    It is highly likely that your model just produces garbage.'
            msg += '\n    Be patient yo, it will get better.'
            self.logger.info(msg)
        else:
            bleu = float(out_parse.group()[6:])

        validation_file = "{}-{}".format(val_trans_out, bleu)
        shutil.copyfile(val_trans_out, validation_file)

        beam_file = "{}-{}".format(val_beam_out, bleu)
        shutil.copyfile(val_beam_out, beam_file)

        # add summaries
        self.bleu_curve = numpy.append(self.bleu_curve, bleu)
        numpy.save(self.bleu_curve_path, self.bleu_curve)

    def evaluate(self, model):
        self.evaluate_perp(model)
        if self.val_by_bleu:
            self.evaluate_bleu(model)

    def _is_valid_to_save(self):
        if self.val_by_bleu:
            if len(self.best_bleus) < self.n_best:
                return None, True
            else:
                min_idx = numpy.argmin(self.best_bleus)
                min_bleu = self.best_bleus[min_idx]
                if min_bleu > self.bleu_curve[-1]:
                    return None, False
                else:
                    return min_idx, True
        else:
            if len(self.best_perps) < self.n_best:
                return None, True
            else:
                max_idx = numpy.argmax(self.best_perps)
                max_perp = self.best_perps[max_idx]
                if max_perp < self.perp_curve[-1]:
                    return None, False
                else:
                    return max_idx, True

    def maybe_save(self, model):
        remove_idx, save_please = self._is_valid_to_save()

        if self.val_by_bleu:
            bleu_score = self.bleu_curve[-1]
            if remove_idx is not None:
                min_bleu = self.best_bleus[remove_idx]
                self.logger.info('Current best bleus: {}'.format(', '.join(map(str, numpy.sort(self.best_bleus)))))
                self.logger.info('Delete {} & use {} instead'.format(min_bleu, bleu_score))
                self.best_bleus = numpy.delete(self.best_bleus, remove_idx)

                # Delete the right checkpoint
                cpkt_path = self.get_cpkt_path(min_bleu)

                if exists(cpkt_path):
                    self.logger.info('Delete {}'.format(cpkt_path))
                    os.remove(cpkt_path)

            if save_please:
                self.logger.info('Save {} to list of best bleu scores'.format(bleu_score))
                self.best_bleus = numpy.append(self.best_bleus, bleu_score)
                cpkt_path = self.get_cpkt_path(bleu_score)
                torch.save(model.state_dict(), cpkt_path)
                self.logger.info('Save new best model to {}'.format(cpkt_path))
                self.logger.info('Best bleu scores so far: {}'.format(', '.join(map(str, numpy.sort(self.best_bleus)))))

            numpy.save(self.best_bleus_path, self.best_bleus)
        else:
            # super redundant but i'm lazy
            perp_score = self.perp_curve[-1]
            if remove_idx is not None:
                max_perp = self.best_perps[remove_idx]
                self.logger.info('Current best perps: {}'.format(', '.join(map(str, numpy.sort(self.best_perps)[::-1]))))
                self.logger.info('Delete {} & use {} instead'.format(max_perp, perp_score))
                self.best_perps = numpy.delete(self.best_perps, remove_idx)

                # Delete the right checkpoint
                cpkt_path = self.get_cpkt_path(max_perp)

                if exists(cpkt_path):
                    self.logger.info('Delete {}'.format(cpkt_path))
                    os.remove(cpkt_path)

            if save_please:
                self.logger.info('Save {} to list of best perp scores'.format(perp_score))
                self.best_perps = numpy.append(self.best_perps, perp_score)
                cpkt_path = self.get_cpkt_path(perp_score)
                torch.save(model.state_dict(), cpkt_path)
                self.logger.info('Save new best model to {}'.format(cpkt_path))
                self.logger.info('Best perp scores so far: {}'.format(', '.join(map(str, numpy.sort(self.best_perps)))))

            numpy.save(self.best_perps_path, self.best_perps)

    def validate_and_save(self, model):
        self.logger.info('Start validation')
        self.evaluate(model)
        self.maybe_save(model)

    def remove_bpe(self, infile, outfile=None):
        if not outfile:
            outfile = infile + '.nobpe'

        open(outfile, 'w').close()
        Popen("sed -r 's/(@@ )|(@@ ?$)//g' < {} > {}".format(infile, outfile), shell=True, stdout=PIPE).communicate()
        return outfile

    def translate(self, model, input_file):
        model.eval()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Very redundant
        best_trans_file = input_file + '.best_trans'
        beam_trans_file = input_file + '.beam_trans'

        num_sents = 0
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    num_sents += 1
        all_best_trans = [''] * num_sents
        all_beam_trans = [''] * num_sents
        with torch.no_grad():
            self.logger.info('Start translating {}'.format(input_file))
            start = time.time()
            count = 0
            for (src_toks, original_idxs) in self.data_manager.get_trans_input(input_file):
                src_toks_cuda = src_toks.to(device)
                rets = model.beam_decode(src_toks_cuda)

                for i, ret in enumerate(rets):
                    probs = ret['probs'].cpu().detach().numpy().reshape([-1])
                    scores = ret['scores'].cpu().detach().numpy().reshape([-1])
                    symbols = ret['symbols'].cpu().detach().numpy()

                    best_trans, beam_trans = self.get_trans(probs, scores, symbols)
                    all_best_trans[original_idxs[i]] = best_trans + '\n'
                    all_beam_trans[original_idxs[i]] = beam_trans + '\n\n'

                    count += 1
                    if count % 100 == 0:
                        self.logger.info('  Translating line {}, average {} seconds/sent'.format(count, (time.time() - start) / count))

        model.train()

        open(best_trans_file, 'w').close()
        open(beam_trans_file, 'w').close()
        with open(best_trans_file, 'w') as ftrans, open(beam_trans_file, 'w') as btrans:
            ftrans.write(''.join(all_best_trans))
            btrans.write(''.join(all_beam_trans))

        self.logger.info('Done translating {}, it takes {} minutes'.format(input_file, float(time.time() - start) / 60.0))
