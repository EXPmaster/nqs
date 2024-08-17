import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sample_mps import sample_mps


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LLMnqs(nn.Module):

    def __init__(self, num_qubits, d_model=256, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=False
            ),
            num_layers=4
        )
        self.src_mask = None
        self.d_model = d_model
        self.seq_len = num_qubits
        self.input_embedding = nn.Embedding(3, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=num_qubits)
        self.output_layer = nn.Linear(d_model, 3)
    
    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def forward(self, x):
        # x: [seq_len, bs]
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            mask = self._generate_square_subsequent_mask(len(x)).to(self.device)
            self.src_mask = mask
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.model(self.positional_encoding(x), mask=self.src_mask, is_causal=True)
        return self.output_layer(x)  # [seq_len, bs, 3]
    
    def fit(
        self,
        meas_bitstrings,
        lr=1e-3,
        epochs=100,
        batch_size=512,
        device='cuda',
        verbose=False
    ):
        self.to(device)
        self.device = device
        self.train()
        # target_state = [mps.to(self.device) for mps in target_state]
        meas_bitstrings = torch.as_tensor(
            meas_bitstrings, dtype=torch.long, device=device
        )
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for ep in range(epochs):
            for idx in range(0, meas_bitstrings.shape[0], batch_size):
                measurement_bitstrings = meas_bitstrings[idx:idx + batch_size]
                # measurement_bitstrings = sample_mps(target_state, batch_size)
                input_bitstrings = measurement_bitstrings[:, :-1]
                input_bitstrings = F.pad(input_bitstrings, (1, 0), value=2)
                target_bitstrings = measurement_bitstrings
                output_logits = self(input_bitstrings.permute(1, 0))
                optimizer.zero_grad()
                loss = criterion(
                    output_logits.view(-1, output_logits.shape[2]),
                    target_bitstrings.view(-1)
                )
                if verbose:
                    print(f'Epoch: {ep}, loss: {loss.item()}')
                loss.backward()
                optimizer.step()

    def sample(self, num_samples=1024):
        self.eval()
        with torch.no_grad():
            input_token = 2 * torch.ones(num_samples, dtype=torch.long, device=self.device)
            input_token = input_token.unsqueeze(0)
            # print(input_token.shape)
            for i in range(self.seq_len):
                predict_prob = self(input_token)
                distr = torch.distributions.Categorical(logits=F.softmax(predict_prob[-1, :, :-1], dim=-1))
                predict_token = distr.sample()
                # predict_token = predict_prob.argmax(-1)
                input_token = torch.cat((input_token, predict_token[None]), 0)
        return input_token.permute(1, 0)[:, 1:]


if __name__ == '__main__':
    import pickle
    
    with open('samples_ising_50q_-83.4122.pkl', 'rb') as f:
        bitstrings = pickle.load(f)
    model = LLMnqs(50)
    model.fit(bitstrings, epochs=100, verbose=True)
    torch.save(model, 'model.pth')
    model = torch.load('model.pth')
    samples = model.sample(2048)
    # compare classical fidelity
    samples = samples.cpu().numpy()
    print(bitstrings)
    bitstrings = bitstrings[:1024]
    bit_prob = bitstrings.mean(axis=0)
    sample_prob = (samples).mean(axis=0)
    print(bit_prob)
    print(sample_prob)
