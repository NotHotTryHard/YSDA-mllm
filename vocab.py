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
        self.tokens = tokens
        # your code here \/

        # your code here /\
        
        assert {bos_token, eos_token, unk_token}.issubset(set(tokens)), \
            f"Missing special tokens. Expected: {bos_token}, {eos_token}, {unk_token}"

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def from_lines(lines, bos="_BOS_", eos="_EOS_", unk='_UNK_'):
        """Create vocabulary from a list of strings."""
        tokens = sorted(set('\n'.join(list(lines)).split()))
        tokens = [t for t in tokens if t not in (bos, eos, unk) and len(t)]
        tokens = [bos, eos, unk] + tokens
        return Vocab(tokens, bos, eos, unk)
    

    def tokenize(self, string):
        """Сonverts string to a list of tokens"""
        # your code here \/

        # your code here /\
        

    def to_matrix(self, lines, max_len=None, dtype=torch.int64, device="cpu"):
        """
        Сonvert variable length token sequences into a fixed size matrix.

        :param lines: list of tokenized strings (space-separated)
        :param max_len: maximum sequence length 
        :returns: tensor [batch_size, max_len]
        """
        # your code here \/

        # your code here /\
        return matrix

    def to_lines(self, matrix, crop=True):
        """
        Convert matrix of token ids into strings

        :param matrix: matrix of token ids, shape=[batch_size, max_len]
        :param crop: if True, crops BOS and EOS from line
        :return:        
        """
        # your code here \/

        # your code here /\
        return lines
    
    def compute_mask(self, input_ix):
        """ compute a boolean mask that equals "1" until first EOS (including that EOS) """
        # your code here \/

        # your code here /\
