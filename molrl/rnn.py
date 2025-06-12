import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .dataset import SMILESVAEDataset
from .vocab import Vocab
from .utils import valid


def load_vocab_from_dict(vocab: dict):
    char2idx = vocab 
    idx2char = {v: k for k, v in char2idx.items()}
    vocab = Vocab(df=None, smiles_col=None, char2idx=char2idx, idx2char=idx2char)
    return vocab

## RNN to RNN VAE
class SMILESVAERNN(pl.LightningModule):
    def __init__(
        self,
        latent_size,
        hidden_dim,
        num_layers,
        vocab,
        bidirectional=True,
        word_dropout=0.5,
        lr=0.001,
        kl_weight=1,
        t_kl_weight=0.0025,
        c_step=1000,
        dropout=0.3,
        annealing=True,
        max_kl_weight=1,
        variance=1,
        emb_dim=32,
        enc_num_layers=2,
        n_cycles_anneal=20,
    ):
        super().__init__()
        if isinstance(vocab, dict):
            vocab = load_vocab_from_dict(vocab)
        
        self.max_kl_weight = max_kl_weight
        self.c_step = c_step
        self.n_cycles_anneal = n_cycles_anneal
        self.t_kl_weight = t_kl_weight
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.enc_num_layers = enc_num_layers
        self.enc_hidden_factor = (2 if bidirectional else 1) * self.enc_num_layers
        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.vocab = vocab
        self.latent_size = latent_size
        self.emb_dim = emb_dim
        self.annealing = annealing

        self.embedding = nn.Embedding(vocab.vocab_size, self.emb_dim, padding_idx=vocab.pad_idx)
        self.encoder = nn.GRU(
            self.emb_dim,
            hidden_dim,
            batch_first=True,
            num_layers=2,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.decoder = nn.GRU(
            self.emb_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=False,
            dropout=dropout,
        )
        self.mu_fc = nn.Linear(hidden_dim * self.enc_hidden_factor, latent_size)
        self.logvar_fc = nn.Linear(hidden_dim * self.enc_hidden_factor, latent_size)

        self.latent2hidden = nn.Linear(latent_size, hidden_dim * self.num_layers)
        self.outputs2vocab = nn.Linear(hidden_dim, vocab.vocab_size)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, reduction='sum')
        self.word_dropout = word_dropout
        self.lr = lr
        self.kl_weight = kl_weight
        print(f'annealing: {self.annealing}')
        self.save_hyperparameters()

    def resize_hidden_encoder(self, h, batch_size):
        # h shape (direction*layers, batch_size, output_dim)
        h = h.transpose(0, 1).contiguous()
        if self.bidirectional or self.num_layers > 1:
            hidden_factor = (2 if self.bidirectional else 1) * self.enc_num_layers
            h = h.view(batch_size, self.hidden_dim * hidden_factor)
            # h shape (batch_size, direction*layers * output_dim)
        else:
            h = h.squeeze()
        return h

    def resize_hidden_decoder(self, h, batch_size):
        # h shape (batch_size, direction*layers*output_dim)
        if self.num_layers > 1:
            # flatten hidden state
            h = h.view(batch_size, self.num_layers, self.hidden_dim)
            h = h.transpose(0, 1).contiguous()
            # h = h.view(self.num_layers, batch_size, self.hidden_dim)
        else:
            h = h.unsqueeze(0)
        return h

    def _cyclical_anneal(self, T, M, t, R=0.7):
        tm = T / M
        mod = (t - 1) % (tm)
        a = mod / tm
        if a > R:
            a = R
        else:
            a = min(1, a)
        return a

    def kl_anneal_function(self, anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(self.max_kl_weight / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return min(self.max_kl_weight, step / x0)
        elif anneal_function == 'cyclical':
            return self._cyclical_anneal(
                self.c_step, self.n_cycles_anneal, t=step, R=self.max_kl_weight
            )

    def mask_inputs(self, x):
        x_mutate = x.clone()
        prob = torch.rand_like(x.float())
        prob[(x_mutate - self.vocab.sos_idx) * (x_mutate - self.vocab.pad_idx) == 0] = 1
        x_mutate[prob < self.word_dropout] = self.vocab.unk_idx
        return x_mutate

    def compute_loss(self, outputs, targets, lengths, logvar, mu):
        targets = targets[:, : torch.max(torch.tensor(lengths)).item()].contiguous().view(-1)
        outputs = outputs.view(-1, outputs.size(2))
        r_loss = self.criterion(outputs, targets)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        kl_loss = kl_loss * self.kl_weight
        return r_loss, kl_loss

    def enc_forward(self, x, lengths):
        batch_size = x.shape[0]

        e = self.embedding(x)

        lengths = torch.tensor(lengths).cpu().numpy()
        x_packed = pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.encoder(x_packed)  # h shape (direction*layers, batch_size, output_dim)
        h = self.resize_hidden_encoder(h, batch_size)
        mu, logvar = self.mu_fc(h), self.logvar_fc(h)
        z = torch.normal(0, 1, size=mu.size())
        z = z.to(self.device)
        std = torch.exp(0.5 * logvar)
        z = z * std + mu
        return z, mu, logvar

    def dec_forward(self, x, z, lengths):
        batch_size = x.shape[0]
        h = self.latent2hidden(z)
        x = self.mask_inputs(x)
        e = self.embedding(x)
        lengths = torch.tensor(lengths).cpu().numpy()
        packed_input = pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        h = self.resize_hidden_decoder(h, batch_size)
        outputs, _ = self.decoder(packed_input, h)
        padded_outputs = pad_packed_sequence(outputs, batch_first=True)[0]
        output_v = self.outputs2vocab(padded_outputs)
        return output_v

    def forward(self, data):
        with_bos, with_eos, lengths = data
        z, mu, logvar = self.enc_forward(with_bos, lengths)
        outputs = self.dec_forward(with_bos, z, lengths)
        return outputs, with_eos, lengths, logvar, mu

    def training_step(self, batch, batch_idx):
        if self.annealing:
            self.kl_weight = self.kl_anneal_function(
                'cyclical', self.trainer.global_step, self.t_kl_weight, self.c_step
            )
        else:
            self.kl_weight = self.max_kl_weight
        outputs = self.forward(batch)
        r_loss, kl_loss = self.compute_loss(*outputs)
        r = {}
        loss = r_loss + kl_loss
        r['loss'] = loss
        r['kl_loss'] = kl_loss
        r['r_loss'] = r_loss
        r['kl_weight'] = self.kl_weight
        r['lr'] = self.lr
        for key in r:
            self.log(f'train_{key}', r[key])
        return r

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        r_loss, kl_loss = self.compute_loss(*outputs)
        r = {}
        loss = r_loss + kl_loss
        r['loss'] = loss
        r['kl_loss'] = kl_loss
        r['r_loss'] = r_loss
        for key in r:
            self.log(f'val_{key}', r[key])
        return r

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def create_model(self, hparams):
        return SMILESVAERNN(**hparams)

    def smiles_to_latent(
        self,
        smiles,
        canonicalize=False,
        num_workers=4,
        add_gaussian=False,
        variance=0.1,
        selfies=False,
        eval_mode=True,
        batch_size=256,
        show_progress=True,
        return_cpu=True,
    ):
        import pandas as pd
        from tqdm import tqdm

        if eval_mode is True:
            self.eval()
        df = pd.DataFrame({'smiles': smiles})
        if selfies is False:
            ds = SMILESVAEDataset(df, smiles_col='smiles', vocab=self.vocab, verbose=False)
        else:
            ds = SelfiesVAEDataset(df, selfies_col='smiles', vocab=self.vocab, verbose=False)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=256, shuffle=False, collate_fn=ds.collate, num_workers=num_workers
        )
        out = list()
        device = next(self.embedding.parameters()).device
        with torch.no_grad():
            for batch in tqdm(loader, leave=False, disable=not show_progress):
                with_bos, _, lengths = batch
                batch_size = with_bos.shape[0]

                e = self.embedding(with_bos.to(device))
                x_packed = pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
                _, h = self.encoder(x_packed)
                h = self.resize_hidden_encoder(h, batch_size)
                mu = self.mu_fc(h)

                # mu = mu.detach().numpy()
                if add_gaussian:
                    z = torch.normal(0, variance, size=mu.size())
                    z = z.to(self.device)
                    z = mu + z
                else:
                    z = mu
                out.append(z)

        out = torch.cat(out)
        if return_cpu:
            out = out.detach().cpu().numpy()

        return out

    def handle_explicit_chars(self, s):
        s = s.replace('Q', 'Cl')
        s = s.replace('X', '[nH]')
        s = s.replace('Y', '[H]')
        s = s.replace('W', 'Br')
        return s

    def reverse_explicit_chars(self, s):
        s = s.replace('Cl', 'Q')
        s = s.replace('[nH]', 'X')
        s = s.replace('[H]', 'Y')
        s = s.replace('Br', 'W')
        return s

    def multinomial_generation(self, total, latent_size=32, max_len=300):
        vseqs = []
        for _ in range(total):
            seq = []
            z = torch.randn([1, self.latent_size])
            # z = torch.normal(0, 1, size=[self.latent_size])
            hidden = self.latent2hidden(z)
            hidden = self.resize_hidden_decoder(hidden, 1).float()
            inputs = torch.tensor([self.vocab.sos_idx])
            inputs = inputs.unsqueeze(1)
            e = self.embedding(inputs)
            for _ in range(max_len):
                outputs, hidden = self.decoder(e, hidden)
                # hidden = self.resize_hidden_decoder(hidden, 1).float()
                output_v = self.outputs2vocab(outputs)
                output_v = output_v.flatten()
                output_v = torch.nn.functional.softmax(output_v, dim=0)
                next_char = torch.multinomial(output_v, 1).item()
                if next_char != self.vocab.eos_idx:
                    inputs = torch.tensor([next_char])
                    inputs = inputs.unsqueeze(1)
                    c = self.vocab.idx2char[next_char]
                    seq.append(c)
                    e = self.embedding(inputs)
                    # print(seq)
                else:
                    seq = ''.join(seq)
                    seq = self.handle_explicit_chars(seq)
                    vseqs.append(seq)
                    break
        return vseqs

    def multinomial_reconstruction(
        self,
        smi,
        total,
        latent_size=32,
        max_len=300,
        argmax=False,
        add_gaussian=False,
        gaussian_variance=0.1,
        selfies=False,
    ):
        device = next(self.decoder.parameters()).device
        vseqs = []
        z = self.smiles_to_latent(
            [smi], add_gaussian=add_gaussian, variance=gaussian_variance, selfies=selfies
        )
        z = torch.tensor(z)
        z = z.to(device)
        print(device)
        for _ in range(total):
            seq = []

            hidden = self.latent2hidden(z).to(device)
            hidden = self.resize_hidden_decoder(hidden, 1).float()
            inputs = torch.tensor([self.vocab.sos_idx])
            inputs = inputs.unsqueeze(1)
            e = self.embedding(inputs)
            for _ in range(max_len):
                outputs, hidden = self.decoder(e, hidden)
                # hidden = self.resize_hidden_decoder(hidden, 1).float()
                output_v = self.outputs2vocab(outputs)
                output_v = output_v.flatten()
                output_v = torch.nn.functional.softmax(output_v, dim=0)
                if argmax is False:
                    next_char = torch.multinomial(output_v, 1).item()
                else:
                    next_char = torch.argmax(output_v).item()
                if next_char != self.vocab.eos_idx:
                    inputs = torch.tensor([next_char]).to(device)
                    inputs = inputs.unsqueeze(1)
                    c = self.vocab.idx2char[next_char]
                    seq.append(c)
                    e = self.embedding(inputs)
                else:
                    seq = ''.join(seq)
                    seq = self.handle_explicit_chars(seq)
                    vseqs.append(seq)
                    break
        return vseqs


# Inhertits the above base class but concatenates the latent space with embedding as decoder input.
class SMILESVAERNNCAT(SMILESVAERNN):
    def __init__(
        self,
        latent_size,
        hidden_dim,
        num_layers,
        vocab,
        bidirectional=True,
        word_dropout=0.5,
        lr=0.001,
        kl_weight=1,
        t_kl_weight=0.0025,
        c_step=1000,
        dropout=0.3,
        annealing=True,
        max_kl_weight=1,
        variance=1,
        emb_dim=32,
        enc_num_layers=2,
    ):
        super().__init__(
            latent_size,
            hidden_dim,
            num_layers,
            vocab,
            bidirectional,
            word_dropout,
            lr,
            kl_weight,
            t_kl_weight,
            c_step,
            dropout,
            annealing=annealing,
            max_kl_weight=max_kl_weight,
            variance=variance,
            emb_dim=emb_dim,
            enc_num_layers=enc_num_layers,
        )
        self.encoder = nn.GRU(
            self.emb_dim,
            hidden_dim,
            batch_first=True,
            num_layers=enc_num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.decoder = nn.GRU(
            self.emb_dim + self.latent_size,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=False,
            dropout=dropout,
        )

    def dec_forward(self, x, z, lengths):
        batch_size = x.shape[0]

        x = self.mask_inputs(x)
        e = self.embedding(x)
        z_0 = z.unsqueeze(1).repeat(1, e.size(1), 1)
        x_input = torch.cat([e, z_0], dim=-1)
        lengths = torch.tensor(lengths).cpu().numpy()
        packed_input = pack_padded_sequence(
            x_input, lengths, batch_first=True, enforce_sorted=False
        )
        h = self.latent2hidden(z)
        h = self.resize_hidden_decoder(h, batch_size)
        outputs, _ = self.decoder(packed_input, h)
        padded_outputs = pad_packed_sequence(outputs, batch_first=True)[0]
        output_v = self.outputs2vocab(padded_outputs)
        return output_v

    def multinomial_generation(self, total, latent_size=32, max_len=300):
        vseqs = []
        for _ in range(total):
            seq = []
            z = torch.randn([1, self.latent_size]).to(self.device)
            # z = torch.normal(0, 1, size=[1, self.latent_size])
            hidden = self.latent2hidden(z)
            hidden = self.resize_hidden_decoder(hidden, 1).float()
            inputs = torch.tensor([self.vocab.sos_idx])
            inputs = inputs.unsqueeze(1).to(device=self.device)
            e = self.embedding(inputs)
            z_0 = z.unsqueeze(1).repeat(1, e.size(1), 1)
            x_input = torch.cat([e, z_0], dim=-1)
            for _ in range(max_len):
                outputs, hidden = self.decoder(x_input, hidden)
                # hidden = self.resize_hidden_decoder(hidden, 1).float()
                output_v = self.outputs2vocab(outputs)
                output_v = output_v.flatten()
                output_v = torch.nn.functional.softmax(output_v, dim=0)
                # next_char = torch.argmax(output_v).item()
                next_char = torch.multinomial(output_v, 1).item()
                if next_char != self.vocab.eos_idx:
                    inputs = torch.tensor([next_char])
                    inputs = inputs.unsqueeze(1).to(self.device)
                    c = self.vocab.idx2char[next_char]
                    seq.append(c)
                    e = self.embedding(inputs)
                    z_0 = z.unsqueeze(1).repeat(1, e.size(1), 1)
                    x_input = torch.cat([e, z_0], dim=-1)
                else:
                    seq = ''.join(seq)
                    seq = self.handle_explicit_chars(seq)
                    vseqs.append(seq)
                    break
        return vseqs

    def multinomial_reconstruction(
        self,
        smi,
        total=100,
        max_len=300,
        argmax=False,
        return_prob=False,
        add_gaussian=False,
        gaussian_variance=0.1,
        selfies=False,
    ):
        """
        Reconstruct SMILES strings from latent vectors using multinomial sampling.

        Parameters:
        -----------
        smi: str
            SMILES string.
        total: int
            Number of SMILES strings to generate per latent vector.
        max_len: int
            Maximum length of the SMILES string.
        argmax: bool
            Whether to use argmax sampling.
        return_prob: bool
            Whether to return the log probability of the generated SMILES string.
        perturb: bool
            Whether to perturb the latent vector.
        add_gaussian: bool
            Whether to add Gaussian noise to the latent vector.
        gaussian_variance: float
            Variance of the Gaussian noise.
        selfies: bool
            Whether to decode to selfies strings.

        Returns:
        --------
        list
            List of generated SMILES strings.
        """
        from .utils import valid
        from rdkit import Chem

        if selfies is False:
            smi = Chem.MolFromSmiles(smi)
            smi = Chem.MolToSmiles(smi, isomericSmiles=False)
        vseqs = []
        device = next(self.decoder.parameters()).device
        z = self.smiles_to_latent(
            [smi], add_gaussian=add_gaussian, variance=gaussian_variance, selfies=selfies
        )
        z = torch.tensor(z)
        z = z.reshape([1, self.latent_size])
        z = z.to(device)

        probs = []
        for _ in range(total):
            seq = []
            s_prob = []
            hidden = self.latent2hidden(z).to(device)
            hidden = self.resize_hidden_decoder(hidden, 1).float()
            inputs = torch.tensor([self.vocab.sos_idx])
            inputs = inputs.unsqueeze(1)
            e = self.embedding(inputs)
            z_0 = z.unsqueeze(1).repeat(1, e.size(1), 1)
            x_input = torch.cat([e, z_0], dim=-1)
            for _ in range(max_len):
                outputs, hidden = self.decoder(x_input, hidden)
                # hidden = self.resize_hidden_decoder(hidden, 1).float()
                output_v = self.outputs2vocab(outputs)
                output_v = output_v.flatten()
                output_v = torch.nn.functional.softmax(output_v, dim=0)
                if argmax is False:
                    next_char = torch.multinomial(output_v, 1).item()
                    p = output_v[next_char].item()
                    s_prob.append(p)
                else:
                    next_char = torch.argmax(output_v).item()
                if next_char != self.vocab.eos_idx:
                    inputs = torch.tensor([next_char])
                    inputs = inputs.unsqueeze(1)
                    c = self.vocab.idx2char[next_char]
                    seq.append(c)
                    e = self.embedding(inputs)
                    z_0 = z.unsqueeze(1).repeat(1, e.size(1), 1)
                    x_input = torch.cat([e, z_0], dim=-1)
                else:
                    seq = ''.join(seq)
                    seq = self.handle_explicit_chars(seq)

                    if selfies is False:
                        if valid(seq):
                            probs.append(np.log(np.product(s_prob)))
                            vseqs.append(seq)
                    else:
                        vseqs.append(seq)
                    break

        if return_prob is False:
            return vseqs
        else:
            return vseqs, probs

    def latent_to_smiles(
        self,
        z: torch.Tensor,
        total=1,
        max_len=300,
        argmax=True,
        return_prob=False,
        selfies=False,
    ):
        """
        Latent to SMILES, recommend to use batched version for large number of samples!!

        Parameters:
        -----------
        z: torch.Tensor
            Latent vector.
        total: int
            Number of SMILES strings to generate per latent vector.
        max_len: int
            Maximum length of the SMILES string.
        argmax: bool
            Whether to use argmax sampling.
        return_prob: bool
            Whether to return the log probability of the generated SMILES string.
        selfies: bool
            Whether to decode to selfies strings.

        Returns:
        --------
        list
            List of generated SMILES strings.
        """
        vseqs, probs = [], []
        device = next(self.decoder.parameters()).device
        z = z.reshape([1, self.latent_size]).to(device)
        for _ in range(total):
            seq = []
            s_prob = []
            hidden = self.latent2hidden(z).to(device)
            hidden = self.resize_hidden_decoder(hidden, 1).float()
            inputs = torch.tensor([self.vocab.sos_idx], device=device)
            inputs = inputs.unsqueeze(1)
            e = self.embedding(inputs)
            z_0 = z.unsqueeze(1).repeat(1, e.size(1), 1)
            x_input = torch.cat([e, z_0], dim=-1)
            for _ in range(max_len):
                outputs, hidden = self.decoder(x_input, hidden)
                # hidden = self.resize_hidden_decoder(hidden, 1).float()
                output_v = self.outputs2vocab(outputs)
                output_v = output_v.flatten()
                output_v = torch.nn.functional.softmax(output_v, dim=0)
                if argmax is False:
                    next_char = torch.multinomial(output_v, 1).item()
                    p = output_v[next_char].item()
                    s_prob.append(p)
                else:
                    next_char = torch.argmax(output_v).item()
                if next_char != self.vocab.eos_idx:
                    inputs = torch.tensor([next_char], device=device)
                    inputs = inputs.unsqueeze(1)
                    c = self.vocab.idx2char[next_char]
                    seq.append(c)
                    e = self.embedding(inputs)
                    z_0 = z.unsqueeze(1).repeat(1, e.size(1), 1)
                    x_input = torch.cat([e, z_0], dim=-1)
                else:
                    seq = ''.join(seq)
                    seq = self.handle_explicit_chars(seq)
                    if selfies is False:
                        if valid(seq):
                            probs.append(np.log(np.product(s_prob)))
                            vseqs.append(seq)
                    else:
                        probs.append(np.log(np.product(s_prob)))
                        vseqs.append(seq)
                    break

        if len(vseqs) == 0:
            vseqs = [None]
            probs = [0]
        # sort smiles based on probabilities
        prob_seqs = list(zip(probs, vseqs))
        prob_seqs.sort(key=lambda x: x[0], reverse=True)
        probs, vseqs = zip(*prob_seqs)
        if return_prob is False:
            return vseqs
        else:
            return vseqs, probs

    def multinomial_generation_batched(self, total, max_len=300, batch_size=256):
        """
        Generate SMILES strings in batches.

        Parameters:
        -----------
        total: int
            Number of SMILES strings to generate.
        max_len: int
            Maximum length of the SMILES string.
        batch_size: int
            Batch size.

        Returns:
        --------
        list
            List of generated SMILES strings.
        """
        all_generated = []
        total_generated = 0
        for _ in range(total // batch_size):
            print('batch: ', _)

            latents = torch.randn([batch_size, self.latent_size])
            generated = self.latent_to_smiles_batched(latents, max_len=max_len)
            all_generated.extend(generated)
            total_generated += len(generated)
            print('total generated: ', total_generated)
        return all_generated

    def multinomial_reconstruction_batched(
        self,
        smiles,
        total=1,
        max_len=300,
        argmax=False,
        return_prob=False,
        selfies=False,
        add_gaussian=False,
        gaussian_variance=0.1,
    ):
        """
        Reconstruction of SMILES strings in batches.

        Parameters:
        -----------
        smiles: list
            List of SMILES strings.
        total: int
            Number of SMILES strings to generate per latent vector.
        max_len: int
            Maximum length of the SMILES string.
        argmax: bool
            Whether to use argmax sampling.
        return_prob: bool
            Whether to return the log probability of the generated SMILES string.
        selfies: bool
            Whether to decode to selfies strings.

        Returns:
        --------
        list
            List of generated SMILES strings.
        """
        latents = self.smiles_to_latent(smiles)
        latents = torch.tensor(latents)
        all_generated = []
        total_generated = 0
        for i in range(0, len(latents), 256):
            print('batch: ', i)
            batch_latents = latents[i : i + 256]
            if add_gaussian:
                noise = torch.normal(0, gaussian_variance, size=batch_latents.size())
                noise = noise.to(self.device)
                batch_latents = batch_latents + noise

            generated = self.latent_to_smiles_batched(batch_latents, max_len=max_len, argmax=argmax)
            all_generated.extend(generated)
            total_generated += len(generated)
            print('total generated: ', total_generated)
        return all_generated

    def latent_to_smiles_batched(
        self,
        z_batch: torch.Tensor,
        total=1,
        max_len=300,
        argmax=True,
        return_prob=False,
        selfies=False,
        return_hidden=False,
    ):
        """
        Generate SMILES strings from a batch of latent vectors.

        Parameters:
        -----------
        z_batch: torch.Tensor
            Batch of latent vectors.
        total: int
            Number of SMILES strings to generate per latent vector.
        max_len: int
            Maximum length of the SMILES string.
        argmax: bool
            Whether to use argmax sampling.
        return_prob: bool
            Whether to return the log probability of the generated SMILES string.
        selfies: bool
            Whether to decode to selfies strings.
        return_hidden: bool
            Whether to return the hidden states.

        Returns:
        --------
        list
            List of generated SMILES strings.

        """
        device = next(self.decoder.parameters()).device
        batch_size = z_batch.size(0)
        z = z_batch.to(device)

        vseqs = []
        probs = []
        seqs = [[] for _ in range(batch_size)]
        s_probs = [[] for _ in range(batch_size)]
        for _ in range(total):
            hidden = self.latent2hidden(z).to(device)
            hidden = self.resize_hidden_decoder(hidden, batch_size).float()

            inputs = torch.tensor([self.vocab.sos_idx] * batch_size, device=device).unsqueeze(1)
            e = self.embedding(inputs)
            z_0 = z.unsqueeze(1).repeat(1, e.size(1), 1)

            x_input = torch.cat([e, z_0], dim=-1)

            for _ in range(max_len):
                outputs, hidden = self.decoder(x_input, hidden)

                # Process outputs for each sample in the batch
                output_v = self.outputs2vocab(outputs)
                output_v = output_v.view(batch_size, -1)
                output_v = F.softmax(output_v, dim=-1)
                if argmax is False:
                    next_chars = []
                    for i in range(len(output_v)):
                        next_char = torch.multinomial(output_v[i], 1)
                        next_chars.append(next_char)
                    next_chars = torch.tensor(next_chars, device=device)
                else:
                    next_chars = torch.argmax(output_v, dim=-1)

                # Update sequences and inputs for each sample in the batch
                new_inputs = next_chars.clone().unsqueeze(1)
                c = [self.vocab.idx2char[idx.item()] for idx in next_chars]
                for i in range(batch_size):
                    seqs[i].append(c[i])

                e = self.embedding(new_inputs)
                z_0 = z.unsqueeze(1).repeat(1, e.size(1), 1)
                x_input = torch.cat([e, z_0], dim=-1)

            for i in range(batch_size):
                seq = ''.join(seqs[i])
                seq = self.handle_explicit_chars(seq)
                try:
                    seq = seq[0 : seq.index(self.vocab.eos_token)]
    
                    if selfies is False:
                        if valid(seq):
                            vseqs.append(seq)
                        else:
                            probs.append(None)
                            vseqs.append(None)
                    else:
                        vseqs.append(seq)
                except Exception:
                    vseqs.append(None)

        if len(vseqs) == 0:
            vseqs = [None]

        return vseqs

