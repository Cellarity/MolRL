import torch
import numpy as np
from .vocab import Vocab
from torch.nn.utils.rnn import pad_sequence

## RNN --> RNN Dataset simple character wise tokenization
class SMILESVAEDataset:
    """Canonical to Canonical Dataset."""

    def __init__(self, df, smiles_col, vocab, verbose=True, selfies=False):
        self.smiles_col = smiles_col
        self.vocab = vocab
        self.sos_token = self.vocab.sos_token
        self.eos_token = self.vocab.eos_token
        self.pad_token = self.vocab.pad_token
        self.unk_token = self.vocab.unk_token
        self.char2idx = vocab.char2idx
        self.idx2char = vocab.idx2char
        self.verbose = verbose
        self.selfies = selfies
        self.tokenize(df)

    def __getitem__(self, idx):
        seq = self.tokens[idx]
        with_bos = torch.tensor([self.char2idx[self.sos_token]] + seq, dtype=torch.long)
        with_eos = torch.tensor(seq + [self.char2idx[self.eos_token]], dtype=torch.long)
        return (with_bos, with_eos)

    def __len__(self):
        return len(self.tokens)

    def collate(self, samples):
        with_bos, with_eos = list(zip(*samples))
        lengths = np.array([len(x) for x in with_bos])
        with_bos = torch.nn.utils.rnn.pad_sequence(
            with_bos, padding_value=self.char2idx[self.pad_token], batch_first=True
        )
        with_eos = torch.nn.utils.rnn.pad_sequence(
            with_eos, padding_value=self.char2idx[self.pad_token], batch_first=True
        )
        return with_bos, with_eos, lengths

    def get_vocab(self):
        return Vocab(self.char2idx, self.idx2char)

    def tokenize(self, df):
        from tqdm import tqdm
        import selfies as sf

        if self.verbose:
            print('tokenizing..')
        all_smi = df[self.smiles_col].values
        self.tokens = []
        for _, smi in enumerate(tqdm(all_smi, disable=bool(int(self.verbose) * -1))):
            if self.selfies is False:
                smi = smi.replace('Cl', 'Q')
                smi = smi.replace('Br', 'W')
                smi = smi.replace('[nH]', 'X')
                smi = smi.replace('[H]', 'Y')
            if self.selfies:
                smi = sf.encoder(smi)
                smi = sf.split_selfies(smi)
            t = [self.char2idx[i] for i in smi]
            self.tokens.append(t)
        if self.verbose:
            print('done..')