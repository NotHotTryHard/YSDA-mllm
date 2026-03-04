import numpy as np
import torch
import torch.nn.functional as F


class Vocab:
    """
    Vocabulary that maps strings to token ids and vice versa.
    Special tokens: <BOS>, <EOS>, <UNK>
    """

    def __init__(
        self, 
        tokens,
        bos_token="_BOS_",
        eos_token="_EOS_",
        unk_token="_UNK_"
    ):
        if unk_token not in tokens:
            tokens = [unk_token] + tokens
        if eos_token not in tokens:
            tokens = [eos_token] + tokens
        if bos_token not in tokens:
            tokens = [bos_token] + tokens

        self.mapper = {token : idx for (idx, token) in enumerate(tokens)}
        self.unmapper = {idx : token for token, idx in self.mapper.items()}
        
        self.bos_idx = self.mapper[bos_token]
        self.eos_idx = self.mapper[eos_token]
        self.unk_idx = self.mapper[unk_token]
        
        assert {bos_token, eos_token, unk_token}.issubset(set(tokens)), \
            f"Missing special tokens. Expected: {bos_token}, {eos_token}, {unk_token}"

    def __len__(self):
        return len(self.mapper)

    @staticmethod
    def from_lines(lines, bos="_BOS_", eos="_EOS_", unk='_UNK_'):
        """Create vocabulary from a list of strings."""
        tokens = sorted(set('\n'.join(list(lines)).split()))
        tokens = [t for t in tokens if t not in (bos, eos, unk) and len(t)]
        tokens = [bos, eos, unk] + tokens
        return Vocab(tokens, bos, eos, unk)

    def tokenize(self, string):
        """Сonverts string to a list of tokens"""
        tokenized_string = [self.bos_idx]
        for token in string.split():
            if token in self.mapper:
                tokenized_string.append(self.mapper[token])
            else:
                tokenized_string.append(self.unk_idx)
        tokenized_string.append(self.eos_idx)
        return tokenized_string
        

    def to_matrix(self, lines, max_len=None, dtype=torch.int64, device="cpu"):
        """
        Сonvert variable length token sequences into a fixed size matrix.

        :param lines: list of tokenized strings (space-separated)
        :param max_len: maximum sequence length 
        :returns: tensor [batch_size, max_len]
        """

        if isinstance(lines, np.ndarray):
            lines = lines.tolist()

        if not lines:
            max_len = max_len or 0
            return torch.empty((0, max_len), dtype=dtype, device=device)

        if isinstance(lines[0], str):
            token_seqs = [self.tokenize(line) for line in lines]
        else:
            token_seqs = lines

        if max_len is None:
            max_len = max(len(seq) for seq in token_seqs) if token_seqs else 0

        tensors = []
        for seq in token_seqs:
            t = torch.tensor(seq, dtype=dtype, device=device)

            if t.numel() > max_len:
                t = t[:max_len]
                #t[-1] = self.eos_idx

            elif t.numel() < max_len:
                t = F.pad(t, (0, max_len - t.numel()), value=self.eos_idx)

            tensors.append(t)

        return torch.stack(tensors, dim=0)

    def to_lines(self, matrix, crop=True):
        """
        Convert matrix of token ids into strings

        :param matrix: matrix of token ids, shape=[batch_size, max_len]
        :param crop: if True, crops BOS and EOS from line
        :return:        
        """

        lines = []
        for row in matrix.tolist():
            symbols = []

            for token in row:
                symb = self.unmapper.get(token, self.unmapper.get(self.unk_idx))

                if crop and token == self.bos_idx:
                    continue

                if token == self.eos_idx:
                    if not crop:
                        symbols.append(symb)
                    break

                symbols.append(symb)

            lines.append(" ".join(symbols))

        return lines
    
    def compute_mask(self, input_ix):
        """ compute a boolean mask that equals "1" until first EOS (including that EOS) """
        eos_mask = (input_ix == self.eos_idx)
        eos_cumsum = eos_mask.to(torch.int64).cumsum(dim=1)

        mask = (eos_cumsum == 0) | ((eos_cumsum == 1) & eos_mask)
        return mask
