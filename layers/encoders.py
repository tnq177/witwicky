import torch
from torch import nn
import torch.nn.functional as F

from layers import Attention, PositionWiseFeedForward


class Encoder(nn.Module):
    """Self-attention Encoder"""
    def __init__(self, num_layers, num_heads, embed_dim, ff_dim, dropout=0., norm_in=True):
        super(Encoder, self).__init__()
        self.self_atts = nn.ModuleList([])
        self.pos_ffs = nn.ModuleList([])
        self.lnorms = nn.ModuleList([])
        for i in range(num_layers):
            self.self_atts.append(Attention(embed_dim, num_heads, dropout=dropout))
            self.pos_ffs.append(PositionWiseFeedForward(embed_dim, ff_dim, dropout=dropout))
            self.lnorms.append(nn.ModuleList([nn.LayerNorm(embed_dim, eps=1e-6) for _ in range(2)]))

        self.last_lnorm = nn.LayerNorm(embed_dim, eps=1e-6) if norm_in else None
        self.dropout = dropout
        self.num_layers = num_layers

    def maybe_layernorm(self, x, lnorm, normalize):
        if normalize:
            return lnorm(x)
        else:
            return x

    def forward(self, src_inputs, src_mask):
        norm_in = self.last_lnorm is not None

        x = F.dropout(src_inputs, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            self_att = self.self_atts[i]
            pos_ff = self.pos_ffs[i]
            lnorms = self.lnorms[i]

            residual = x
            x = self.maybe_layernorm(x, lnorms[0], norm_in)
            x, _ = self_att(q=x, k=x, v=x, mask=src_mask)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = self.maybe_layernorm(x, lnorms[0], not norm_in)

            residual = x
            x = self.maybe_layernorm(x, lnorms[1], norm_in)
            x = pos_ff(x)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = self.maybe_layernorm(x, lnorms[1], not norm_in)

        return self.maybe_layernorm(x, self.last_lnorm, norm_in)


class Decoder(nn.Module):
    """Self-attention Decoder"""
    def __init__(self, num_layers, num_heads, embed_dim, ff_dim, dropout=0., norm_in=True):
        super(Decoder, self).__init__()
        self.self_atts = nn.ModuleList([])
        self.enc_dec_atts = nn.ModuleList([])
        self.pos_ffs = nn.ModuleList([])
        self.lnorms = nn.ModuleList([])
        for i in range(num_layers):
            self.self_atts.append(Attention(embed_dim, num_heads, dropout=dropout))
            self.enc_dec_atts.append(Attention(embed_dim, num_heads, dropout=dropout))
            self.pos_ffs.append(PositionWiseFeedForward(embed_dim, ff_dim, dropout=dropout))
            self.lnorms.append(nn.ModuleList([nn.LayerNorm(embed_dim, eps=1e-6) for _ in range(3)]))

        self.last_lnorm = nn.LayerNorm(embed_dim, eps=1e-6) if norm_in else None
        self.dropout = dropout
        self.num_layers = num_layers

    def maybe_layernorm(self, x, lnorm, normalize):
        if normalize:
            return lnorm(x)
        else:
            return x

    def forward(self, trg_inputs, trg_mask, encoder_out, encoder_mask):
        norm_in = self.last_lnorm is not None

        x = F.dropout(trg_inputs, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            self_att = self.self_atts[i]
            enc_dec_att = self.enc_dec_atts[i]
            pos_ff = self.pos_ffs[i]
            lnorms = self.lnorms[i]

            residual = x
            x = self.maybe_layernorm(x, lnorms[0], norm_in)
            x, _ = self_att(q=x, k=x, v=x, mask=trg_mask)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = self.maybe_layernorm(x, lnorms[0], not norm_in)

            residual = x
            x = self.maybe_layernorm(x, lnorms[1], norm_in)
            x, _ = enc_dec_att(q=x, k=encoder_out, v=encoder_out, mask=encoder_mask)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = self.maybe_layernorm(x, lnorms[1], not norm_in)

            residual = x
            x = self.maybe_layernorm(x, lnorms[2], norm_in)
            x = pos_ff(x)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = self.maybe_layernorm(x, lnorms[2], not norm_in)

        return self.maybe_layernorm(x, self.last_lnorm, norm_in)

    def beam_step(self, inp, cache):
        norm_in = self.last_lnorm is not None
        bsz, beam_size = cache['encoder_mask'].size()[:2]
        cache['encoder_mask'] = cache['encoder_mask'].reshape(bsz * beam_size, 1, 1, -1)
        for i in range(self.num_layers):
            seq_len = cache[i]['enc_dec_k'].size()[2]
            cache[i]['enc_dec_k'] = cache[i]['enc_dec_k'].reshape(bsz * beam_size, seq_len, -1)
            cache[i]['enc_dec_v'] = cache[i]['enc_dec_v'].reshape(bsz * beam_size, seq_len, -1)

            if cache[i]['self_att']['k'] is not None:
                seq_len = cache[i]['self_att']['k'].size()[2]
                cache[i]['self_att']['k'] = cache[i]['self_att']['k'].reshape(bsz * beam_size, seq_len, -1)
                cache[i]['self_att']['v'] = cache[i]['self_att']['v'].reshape(bsz * beam_size, seq_len, -1)

        x = inp # [bsz x beam, 1, D]
        for i in range(self.num_layers):
            self_att = self.self_atts[i]
            enc_dec_att = self.enc_dec_atts[i]
            pos_ff = self.pos_ffs[i]
            lnorms = self.lnorms[i]

            residual = x
            x = self.maybe_layernorm(x, lnorms[0], norm_in)
            q, k, v = self_att.linear_projection(x, x, x)
            if cache[i]['self_att']['k'] is not None:
                k = torch.cat((cache[i]['self_att']['k'], k), 1)
                v = torch.cat((cache[i]['self_att']['v'], v), 1)

            cache[i]['self_att']['k'] = k
            cache[i]['self_att']['v'] = v

            x, _ = self_att(q=q, k=k, v=v, mask=None, do_proj=False)
            x = residual + x
            x = self.maybe_layernorm(x, lnorms[0], not norm_in)

            residual = x
            x = self.maybe_layernorm(x, lnorms[1], norm_in)
            q = enc_dec_att.in_proj_q(x)
            k = cache[i]['enc_dec_k']
            v = cache[i]['enc_dec_v']
            x, _ = enc_dec_att(q=q, k=k, v=v, mask=cache['encoder_mask'], do_proj=False)
            x = residual + x
            x = self.maybe_layernorm(x, lnorms[1], not norm_in)

            residual = x
            x = self.maybe_layernorm(x, lnorms[2], norm_in)
            x = pos_ff(x)
            x = residual + x
            x = self.maybe_layernorm(x, lnorms[2], not norm_in)

        x = self.maybe_layernorm(x, self.last_lnorm, norm_in)

        cache['encoder_mask'] = cache['encoder_mask'].reshape(bsz, beam_size, 1, 1, -1)
        for i in range(self.num_layers):
            seq_len = cache[i]['enc_dec_k'].size()[1]
            cache[i]['enc_dec_k'] = cache[i]['enc_dec_k'].reshape(bsz, beam_size, seq_len, -1)
            cache[i]['enc_dec_v'] = cache[i]['enc_dec_v'].reshape(bsz, beam_size, seq_len, -1)

            seq_len = cache[i]['self_att']['k'].size()[1]
            cache[i]['self_att']['k'] = cache[i]['self_att']['k'].reshape(bsz, beam_size, seq_len, -1)
            cache[i]['self_att']['v'] = cache[i]['self_att']['v'].reshape(bsz, beam_size, seq_len, -1)

        return x

    def beam_decode(self, encoder_out, encoder_mask, get_input_fn, logprob_fn, bos_id, eos_id, max_len, beam_size=4, alpha=-1):
        # first step, beam=1
        batch_size = encoder_out.size()[0]
        inp = get_input_fn(torch.tensor([bos_id] * batch_size).reshape(batch_size, 1), 0) # [bsz, 1, D]
        cache = {'encoder_mask': encoder_mask.unsqueeze_(1)} # [bsz, beam, 1, 1, length]
        for i in range(self.num_layers):
            cache[i] = {'self_att': {'k': None, 'v': None}}
            cache[i]['enc_dec_k'] = self.enc_dec_atts[i].in_proj_k(encoder_out).unsqueeze_(1)
            cache[i]['enc_dec_v'] = self.enc_dec_atts[i].in_proj_v(encoder_out).unsqueeze_(1)

        y = self.beam_step(inp, cache).squeeze_(1) # [bsz, D]
        probs = logprob_fn(y) # [bsz, V]
        probs[:, eos_id] = float('-inf') # no <eos> now
        all_probs, symbols = torch.topk(probs, beam_size, dim=-1) # ([bsz, beam], [bsz, beam])

        last_probs = all_probs.reshape(batch_size, beam_size)
        last_scores = last_probs.clone()
        all_symbols = symbols.reshape(batch_size, beam_size, 1)

        cache['encoder_mask'] = encoder_mask.expand(-1, beam_size, -1, -1, -1)
        for i in range(self.num_layers):
            cache[i]['self_att']['k'] = cache[i]['self_att']['k'].expand(-1, beam_size, -1, -1)
            cache[i]['self_att']['v'] = cache[i]['self_att']['v'].expand(-1, beam_size, -1, -1)
            cache[i]['enc_dec_k'] = cache[i]['enc_dec_k'].expand(-1, beam_size, -1, -1)
            cache[i]['enc_dec_v'] = cache[i]['enc_dec_v'].expand(-1, beam_size, -1, -1)

        num_classes = probs.size()[-1]
        not_eos_mask = (torch.arange(num_classes).reshape(1, -1) != eos_id).type(encoder_mask.type())
        maximum_length = max_len.max().item()
        ret = [None] * batch_size
        batch_idxs = torch.arange(batch_size)
        for time_step in range(1, maximum_length + 1):
            surpass_length = (max_len < time_step) + (time_step == maximum_length)
            finished_decoded = torch.sum(all_symbols[:, :, -1] == eos_id, -1) == beam_size
            finished_sents = ((surpass_length + finished_decoded) >= 1).type(encoder_mask.type())
            if finished_sents.any():
                for j in range(finished_sents.size()[0]):
                    if finished_sents[j]:
                        ret[batch_idxs[j]] = {
                            'symbols': all_symbols[j].clone(),
                            'probs': last_probs[j].clone(),
                            'scores': last_scores[j].clone()
                        }

                all_symbols = all_symbols[~finished_sents]
                last_probs = last_probs[~finished_sents]
                last_scores = last_scores[~finished_sents]
                max_len = max_len[~finished_sents]
                batch_idxs = batch_idxs[~finished_sents]
                cache['encoder_mask'] = cache['encoder_mask'][~finished_sents]
                for i in range(self.num_layers):
                    cache[i]['self_att']['k'] = cache[i]['self_att']['k'][~finished_sents]
                    cache[i]['self_att']['v'] = cache[i]['self_att']['v'][~finished_sents]
                    cache[i]['enc_dec_k'] = cache[i]['enc_dec_k'][~finished_sents]
                    cache[i]['enc_dec_v'] = cache[i]['enc_dec_v'][~finished_sents]

            if finished_sents.all():
                break

            bsz = all_symbols.size()[0]
            last_symbols = all_symbols[:, :, -1]
            inps = get_input_fn(last_symbols, time_step).reshape(bsz * beam_size, -1).unsqueeze_(1) # [bsz x beam, 1, D]
            ys = self.beam_step(inps, cache).squeeze_(1) # [bsz x beam, D]
            probs = logprob_fn(ys) # [bsz x beam, V]
            last_probs = last_probs.reshape(-1, 1) # [bsz x beam, 1]
            last_scores = last_scores.reshape(-1, 1)
            length_penalty = 1.0 if alpha == -1 else (5.0 + time_step + 1.0) ** alpha / 6.0 ** alpha
            finished_mask = (last_symbols.reshape(-1) == eos_id).type(encoder_mask.type())
            beam_probs = probs.clone()
            if finished_mask.any():
                beam_probs[finished_mask] = last_probs[finished_mask].expand(-1, num_classes).masked_fill(not_eos_mask, float('-inf'))
                beam_probs[~finished_mask] = last_probs[~finished_mask] + probs[~finished_mask]
            else:
                beam_probs = last_probs + probs

            beam_scores = beam_probs.clone()
            if finished_mask.any():
                beam_scores[finished_mask] = last_scores[finished_mask].expand(-1, num_classes).masked_fill(not_eos_mask, float('-inf'))
                beam_scores[~finished_mask] = beam_probs[~finished_mask] / length_penalty
            else:
                beam_scores = beam_probs / length_penalty

            beam_probs = beam_probs.reshape(bsz, -1)
            beam_scores = beam_scores.reshape(bsz, -1)
            max_scores, idxs = torch.topk(beam_scores, beam_size, dim=-1) # ([bsz, beam], [bsz, beam])
            parent_idxs = idxs // num_classes
            symbols = (idxs - parent_idxs * num_classes).type(idxs.type()) # [bsz, beam]

            last_probs = torch.gather(beam_probs, -1, idxs)
            last_scores = max_scores
            parent_idxs = parent_idxs + torch.arange(bsz).unsqueeze_(1).type(parent_idxs.type()) * beam_size
            parent_idxs = parent_idxs.reshape(-1)
            all_symbols = all_symbols.reshape(bsz * beam_size, -1)[parent_idxs].reshape(bsz, beam_size, -1)
            all_symbols = torch.cat((all_symbols, symbols.unsqueeze_(-1)), -1)

            for i in range(self.num_layers):
                seq_len = cache[i]['self_att']['k'].size()[2]
                cache[i]['self_att']['k'] = cache[i]['self_att']['k'].reshape(bsz * beam_size, seq_len, -1)[parent_idxs].reshape(bsz, beam_size, seq_len, -1)
                cache[i]['self_att']['v'] = cache[i]['self_att']['v'].reshape(bsz * beam_size, seq_len, -1)[parent_idxs].reshape(bsz, beam_size, seq_len, -1)

        return ret
