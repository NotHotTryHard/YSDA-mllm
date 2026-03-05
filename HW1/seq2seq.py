import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    """A simple encoder-decoder seq2seq model."""

    def __init__(self, inp_voc, out_voc, emb_size=64, hid_size=128):
        super().__init__()
        self.inp_voc, self.out_voc = inp_voc, out_voc
        self.hid_size = hid_size

        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.emb_out = nn.Embedding(len(out_voc), emb_size)
        self.enc0 = nn.GRU(emb_size, hid_size, batch_first=True)

        self.dec_start = nn.Linear(hid_size, hid_size)
        self.dec0 = nn.GRUCell(emb_size, hid_size)
        self.logits = nn.Linear(hid_size, len(out_voc))

    def forward(self, inp, out):
        """Apply model in training mode."""
        initial_state = self.encode(inp)
        return self.decode(initial_state, out)

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state.
        :param inp: matrix of input tokens [batch, time]
        :returns: initial decoder state tensors
        """
        inp_emb = self.emb_inp(inp)

        enc_seq, [last_state_but_not_really] = self.enc0(inp_emb)

        lengths = (inp != self.inp_voc.eos_idx).to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths]

        dec_start = self.dec_start(last_state)
        return [dec_start]

    def decode_step(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits.
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, int vector [batch_size]
        :return: (new_state, logits [batch, n_tokens])
        """
        prev_gru0_state = prev_state[0]

        prev_emb = self.emb_out(prev_tokens)
        next_gru0_state = self.dec0(prev_emb, prev_gru0_state)
        output_logits = self.logits(next_gru0_state)
        new_dec_state = [next_gru0_state]

        return new_dec_state, output_logits

    def decode(self, initial_state, out_tokens, **flags):
        """Iterate over reference tokens (out_tokens) with decode_step."""
        batch_size = out_tokens.shape[0]
        state = initial_state
        device = out_tokens.device

        onehot_bos = F.one_hot(
            torch.full([batch_size], self.out_voc.bos_idx, dtype=torch.int64, device=device),
            num_classes=len(self.out_voc),
        )
        first_logits = torch.log(onehot_bos.to(torch.float32) + 1e-9)

        logits_sequence = [first_logits]
        for i in range(out_tokens.shape[1] - 1):
            state, logits = self.decode_step(state, out_tokens[:, i])
            logits_sequence.append(logits)
        return torch.stack(logits_sequence, dim=1)

    def decode_inference(self, initial_state, max_len=100, **flags):
        """Generate translations from model (greedy version)."""
        batch_size, device = len(initial_state[0]), initial_state[0].device
        state = initial_state
        outputs = [torch.full([batch_size], self.out_voc.bos_idx, dtype=torch.int64, device=device)]
        all_states = [initial_state]

        for i in range(max_len):
            state, logits = self.decode_step(state, outputs[-1])
            outputs.append(logits.argmax(dim=-1))
            all_states.append(state)

        return torch.stack(outputs, dim=1), all_states

    def translate_lines(self, inp_lines, **kwargs):
        """Translate input lines."""
        device = next(self.parameters()).device
        inp = self.inp_voc.to_matrix(inp_lines).to(device)
        initial_state = self.encode(inp)
        out_ids, states = self.decode_inference(initial_state, **kwargs)
        return self.out_voc.to_lines(out_ids.cpu()), states


class AttentionLayer(nn.Module):
    """A layer that computes additive attention response and weights."""

    def __init__(self, name, enc_size, dec_size, hid_size, activ=torch.tanh):
        super().__init__()
        self.name = name
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.hid_size = hid_size
        self.activ = activ

        self.Wenc = nn.Linear(enc_size, hid_size)
        self.Wdec = nn.Linear(dec_size, hid_size)
        self.bias = nn.Linear(hid_size, 1)

    def forward(self, enc, dec, inp_mask):
        """
        Computes attention response and weights.
        :param enc: encoder activations, float32[batch_size, ninp, enc_size]
        :param dec: single decoder state (query), float32[batch_size, dec_size]
        :param inp_mask: mask on enc activations (0 after first eos), float32[batch_size, ninp]
        :returns: attn[batch_size, enc_size], probs[batch_size, ninp]
        """
        dec_proj = self.Wdec(dec)
        enc_proj = self.Wenc(enc)

        scores = self.bias(self.activ(enc_proj + dec_proj.unsqueeze(1))).squeeze(-1)
        scores[~inp_mask.bool()] = float('-inf')
        probs = torch.softmax(scores, dim=-1) 
        weights = probs.unsqueeze(-1)
        weighted_enc = weights * enc
        attn = weighted_enc.sum(dim=1)

        return attn, probs


class AttentiveModel(BasicModel):
    """Translation model that uses attention."""

    def __init__(self, name, inp_voc, out_voc, emb_size=64, hid_size=128, attn_size=128):
        nn.Module.__init__(self)
        self.inp_voc, self.out_voc = inp_voc, out_voc
        self.hid_size = hid_size

        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.emb_out = nn.Embedding(len(out_voc), emb_size)
        self.enc0 = nn.GRU(emb_size, hid_size, batch_first=True, bidirectional=True)
        self.dec_start = nn.Linear(2 * hid_size, hid_size)
        self.dec0 = nn.GRUCell(emb_size + 2 * hid_size, hid_size)
        
        self.attn = AttentionLayer("attn", 2 * hid_size, hid_size, attn_size)
        self.logits = nn.Linear(hid_size + 2 * hid_size, len(out_voc))

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens
        :return: a list of initial decoder state tensors
        """
        # Encode input sequence, create initial decoder states
        inp_emb = self.emb_inp(inp)
        enc_seq, _ = self.enc0(inp_emb)

        #inp_mask = self.inp_voc.compute_mask(inp)  #  зря писал, лмао
        inp_mask = (inp != self.inp_voc.eos_idx).float()
        lengths = inp_mask.to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths]
        dec_start = self.dec_start(last_state)

        # Apply attention layer from initial decoder hidden state
        attn, attn_probs = self.attn(enc_seq, dec_start, inp_mask)

        # Build first state: include
        # * initial states for decoder recurrent layers
        # * encoder sequence and encoder attn mask (for attention)
        # * make sure that last state item is attention probabilities tensor
        first_state = [dec_start, enc_seq, inp_mask, attn, attn_probs]
        return first_state

    def decode_step(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits for next tokens
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch, n_tokens]
        """
        prev_dec, enc_seq, inp_mask, prev_attn, prev_attn_probs = prev_state
        prev_emb = self.emb_out(prev_tokens)
        dec_input = torch.cat([prev_emb, prev_attn], dim=-1)
        new_dec = self.dec0(dec_input, prev_dec)
        
        attn, attn_probs = self.attn(enc_seq, new_dec, inp_mask)
        output_logits = self.logits(torch.cat([new_dec, attn], dim=-1))
        new_state = [new_dec, enc_seq, inp_mask, attn, attn_probs]

        return new_state, output_logits
