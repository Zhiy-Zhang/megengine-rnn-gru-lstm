import math

import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M

# ================================= GRU Implementation ==========================================================


class GRUCell(M.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.ih = M.Linear(input_size, 3 * hidden_size, bias=bias)
        self.hh = M.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            M.init.uniform_(w, -std, std)

    def forward(self, x, hidden):

        x = F.reshape(x, (-1, x.shape[1]))

        gate_x = self.ih(x)
        gate_h = self.hh(hidden)

        i_r, i_i, i_n = F.split(gate_x, 3, axis=1)
        h_r, h_i, h_n = F.split(gate_h, 3, axis=1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class GRU(M.Module):
    """
    An implementation of GRUModule.

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        batch_first=False,
        dropout=0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        self.rnn_cell_list = []

        self.rnn_cell_list.append(GRUCell(self.input_size, self.hidden_size, self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(
                GRUCell(self.hidden_size, self.hidden_size, self.bias)
            )

    def forward(self, input, hx=None):

        if hx is None:
            batch = input.shape[0] if self.batch_first else input.shape[1]
            h0 = F.zeros((self.num_layers, batch, self.hidden_size))
        else:
            h0 = hx

        outs = []
        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        length = input.shape[1] if self.batch_first else input.shape[0]
        for t in range(length):

            for layer in range(self.num_layers):

                if layer == 0:
                    if self.batch_first:
                        hidden_l = self.rnn_cell_list[layer](
                            input[:, t, :], hidden[layer]
                        )
                    else:
                        hidden_l = self.rnn_cell_list[layer](
                            input[t, :, :], hidden[layer]
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1], hidden[layer]
                    )

                if self.dropout and (layer is not self.num_layers - 1):
                    hidden_l = F.dropout(hidden_l, self.dropout)

                hidden[layer] = hidden_l

            outs.append(hidden_l)

        if self.batch_first:
            output = F.stack(outs, axis=1)
        else:
            output = F.stack(outs, axis=0)

        return output


# ================================= LSTM Implementation ==========================================================


class LSTMCell(M.Module):
    """
    An implementation of LSTMCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = M.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = M.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            M.init.uniform_(w, -std, std)

    def forward(self, x, hidden):

        hx, cx = hidden

        x = F.reshape(x, (-1, x.shape[1]))

        gates = self.x2h(x) + self.h2h(hx)

        ingate, forgetgate, cellgate, outgate = F.split(gates, 4, axis=1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = F.mul(cx, forgetgate) + F.mul(ingate, cellgate)

        hy = F.mul(outgate, F.tanh(cy))

        return (hy, cy)


class LSTM(M.Module):
    """
    An implementation of LSTMModule.

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        batch_first=False,
        dropout=0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        self.rnn_cell_list = []

        self.rnn_cell_list.append(
            LSTMCell(self.input_size, self.hidden_size, self.bias)
        )
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(
                LSTMCell(self.hidden_size, self.hidden_size, self.bias)
            )

    def forward(self, input, hx=None):

        if hx is None:
            batch = input.shape[0] if self.batch_first else input.shape[1]
            h0 = F.zeros((self.num_layers, batch, self.hidden_size))
            c0 = F.zeros((self.num_layers, batch, self.hidden_size))
        else:
            h0 = hx[0]
            c0 = hx[1]

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], c0[layer, :, :]))

        length = input.shape[1] if self.batch_first else input.shape[0]
        for t in range(length):

            for layer in range(self.num_layers):

                if layer == 0:
                    inp = input[:, t, :] if self.batch_first else input[t, :, :]
                    hidden_l = self.rnn_cell_list[layer](
                        inp, (hidden[layer][0], hidden[layer][1])
                    )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0], (hidden[layer][0], hidden[layer][1])
                    )
                if self.dropout and (layer is not self.num_layers - 1):
                    hidden_l = (
                        F.dropout(hidden_l[0], self.dropout),
                        F.dropout(hidden_l[1], self.dropout),
                    )

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        if self.batch_first:
            output = F.stack(outs, axis=1)
        else:
            output = F.stack(outs, axis=0)

        return output
