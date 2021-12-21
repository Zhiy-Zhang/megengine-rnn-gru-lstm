import math

import numpy as np
import torch
import torch.nn

import megengine as mge
import megengine.functional as F
import megengine.module as M

from .RNN import GRU, LSTM, GRUCell, LSTMCell


# ===========================================Test GRUCell==============================================
def test_GRUCell_forward():

    input = np.random.randn(6, 3, 10)
    hx = np.random.randn(3, 20)
    m1 = torch.nn.GRUCell(10, 20)
    m2 = GRUCell(10, 20)
    m1.eval()
    m2.eval()

    for m in m1.parameters():
        m.data.fill_(1)
    for m in m2.parameters():
        M.init.fill_(m, 1)

    for i in range(6):
        torch_output = m1(
            torch.tensor(input[i], dtype=torch.float32),
            torch.tensor(hx, dtype=torch.float32),
        )
        mge_output = m2(
            mge.tensor(input[i], dtype=np.float32), mge.tensor(hx, dtype=np.float32)
        )
        np.testing.assert_allclose(
            torch_output.detach().numpy(), mge_output.numpy(), atol=1e-6
        )


def test_GRUCell_backward():

    m1 = torch.nn.GRUCell(10, 20)
    for m in m1.parameters():
        m.data.fill_(1)
    criterion = torch.nn.MSELoss()
    torch_opt = torch.optim.SGD(m1.parameters(), lr=0.0001)
    m1.eval()

    m2 = GRUCell(10, 20)
    for m in m2.parameters():
        M.init.fill_(m, 1)
    mge_opt = mge.optimizer.SGD(m2.parameters(), lr=0.0001)
    mge_gm = mge.autodiff.GradManager().attach(m2.parameters())
    m2.eval()

    input = np.random.randn(6, 3, 10)
    hx = np.random.randn(3, 20)
    targets = np.ones((3, 20), dtype=np.float32)
    for i in range(6):
        mge_w_ih_grad, mge_b_ih_grad, mge_w_hh_grad, mge_b_hh_grad = (
            None,
            None,
            None,
            None,
        )
        torch_w_ih_grad, torch_b_ih_grad, torch_w_hh_grad, torch_b_hh_grad = (
            None,
            None,
            None,
            None,
        )

        torch_output = m1(
            torch.tensor(input[i], dtype=torch.float32),
            torch.tensor(hx, dtype=torch.float32),
        )
        loss = criterion(torch_output, torch.tensor(targets, dtype=torch.float32))
        loss.backward()
        torch_w_ih_grad, torch_b_ih_grad, torch_w_hh_grad, torch_b_hh_grad = (
            m1.weight_ih.grad.numpy().flatten(),
            m1.bias_ih.grad.numpy().flatten(),
            m1.weight_hh.grad.numpy().flatten(),
            m1.bias_hh.grad.numpy().flatten(),
        )
        torch_opt.step()
        torch_opt.zero_grad()

        with mge_gm:
            mge_output = m2(
                mge.tensor(input[i], dtype=np.float32), mge.tensor(hx, dtype=np.float32)
            )
            loss = F.loss.square_loss(mge_output, mge.tensor(targets, dtype=np.float32))
            mge_gm.backward(loss)
            mge_w_ih_grad, mge_b_ih_grad, mge_w_hh_grad, mge_b_hh_grad = (
                m2.ih.weight.grad.numpy().flatten(),
                m2.ih.bias.grad.numpy().flatten(),
                m2.hh.weight.grad.numpy().flatten(),
                m2.hh.bias.grad.numpy().flatten(),
            )
            mge_opt.step().clear_grad()

        np.testing.assert_allclose(mge_w_ih_grad, torch_w_ih_grad, atol=1e-6)
        np.testing.assert_allclose(mge_b_ih_grad, torch_b_ih_grad, atol=1e-6)
        np.testing.assert_allclose(mge_w_hh_grad, torch_w_hh_grad, atol=1e-6)
        np.testing.assert_allclose(mge_b_hh_grad, torch_b_hh_grad, atol=1e-6)


# ===========================================Test LSTM==============================================


def test_LSTMCell_forward():

    input = np.random.randn(6, 3, 10)
    hx = np.random.randn(3, 20)
    cx = np.random.randn(3, 20)
    m1 = torch.nn.LSTMCell(10, 20)
    m2 = LSTMCell(10, 20)
    m1.eval()
    m2.eval()

    for m in m1.parameters():
        m.data.fill_(1)
    for m in m2.parameters():
        M.init.fill_(m, 1)

    for i in range(6):
        torch_param = (
            torch.tensor(hx, dtype=torch.float32),
            torch.tensor(cx, dtype=torch.float32),
        )
        mge_param = (mge.tensor(hx, dtype=np.float32), mge.tensor(cx, dtype=np.float32))
        (torch_output1, torch_output2) = m1(
            torch.tensor(input[i], dtype=torch.float32), torch_param
        )
        (mge_output1, mge_output2) = m2(
            mge.tensor(input[i], dtype=np.float32), mge_param
        )
        np.testing.assert_allclose(
            torch_output1.detach().numpy(), mge_output1.numpy(), atol=1e-6
        )
        np.testing.assert_allclose(
            torch_output2.detach().numpy(), mge_output2.numpy(), atol=1e-6
        )


def test_LSTMCell_backward():

    m1 = torch.nn.LSTMCell(10, 20)
    for m in m1.parameters():
        m.data.fill_(1)
    criterion = torch.nn.MSELoss()
    torch_opt = torch.optim.SGD(m1.parameters(), lr=0.0001)
    m1.eval()

    m2 = LSTMCell(10, 20)
    for m in m2.parameters():
        M.init.fill_(m, 1)
    mge_opt = mge.optimizer.SGD(m2.parameters(), lr=0.0001)
    mge_gm = mge.autodiff.GradManager().attach(m2.parameters())
    m2.eval()

    input = np.random.randn(6, 3, 10)
    hx = np.random.randn(3, 20)
    cx = np.random.randn(3, 20)
    target1 = np.ones((3, 20), dtype=np.float32)
    target2 = np.ones((3, 20), dtype=np.float32)
    for i in range(6):

        mge_w_ih_grad, mge_b_ih_grad, mge_w_hh_grad, mge_b_hh_grad = (
            None,
            None,
            None,
            None,
        )
        torch_w_ih_grad, torch_b_ih_grad, torch_w_hh_grad, torch_b_hh_grad = (
            None,
            None,
            None,
            None,
        )

        torch_param = (
            torch.tensor(hx, dtype=torch.float32),
            torch.tensor(cx, dtype=torch.float32),
        )
        mge_param = (mge.tensor(hx, dtype=np.float32), mge.tensor(cx, dtype=np.float32))
        (torch_output1, torch_output2) = m1(
            torch.tensor(input[i], dtype=torch.float32), torch_param
        )
        loss1 = criterion(torch_output1, torch.tensor(target1, dtype=torch.float32))
        loss2 = criterion(torch_output2, torch.tensor(target2, dtype=torch.float32))
        loss = loss1 + loss2
        loss.backward()
        torch_w_ih_grad, torch_b_ih_grad, torch_w_hh_grad, torch_b_hh_grad = (
            m1.weight_ih.grad.numpy().flatten(),
            m1.bias_ih.grad.numpy().flatten(),
            m1.weight_hh.grad.numpy().flatten(),
            m1.bias_hh.grad.numpy().flatten(),
        )
        torch_opt.step()
        torch_opt.zero_grad()

        with mge_gm:
            (mge_output1, mge_output2) = m2(
                mge.tensor(input[i], dtype=np.float32), mge_param
            )
            loss1 = F.loss.square_loss(
                mge_output1, mge.tensor(target1, dtype=np.float32)
            )
            loss2 = F.loss.square_loss(
                mge_output2, mge.tensor(target2, dtype=np.float32)
            )
            loss = loss1 + loss2
            mge_gm.backward(loss)
            mge_w_ih_grad, mge_b_ih_grad, mge_w_hh_grad, mge_b_hh_grad = (
                m2.x2h.weight.grad.numpy().flatten(),
                m2.x2h.bias.grad.numpy().flatten(),
                m2.h2h.weight.grad.numpy().flatten(),
                m2.h2h.bias.grad.numpy().flatten(),
            )
            mge_opt.step().clear_grad()

        np.testing.assert_allclose(mge_w_ih_grad, torch_w_ih_grad, atol=1e-6)
        np.testing.assert_allclose(mge_b_ih_grad, torch_b_ih_grad, atol=1e-6)
        np.testing.assert_allclose(mge_w_hh_grad, torch_w_hh_grad, atol=1e-6)
        np.testing.assert_allclose(mge_b_hh_grad, torch_b_hh_grad, atol=1e-6)


# ==============================================Test LSTM&GRU Module=============================================


def test_GRU_forward():

    inputs = np.random.randn(3, 6, 10)

    hx = np.random.randn(2, 3, 20)

    m1 = torch.nn.GRU(10, 20, 2, batch_first=True)

    m2 = GRU(10, 20, 2, bias=True, batch_first=True)
    m1.eval()
    m2.eval()

    for m in m1.parameters():
        m.data.fill_(1)
    for m in m2.parameters():
        M.init.fill_(m, 1)

    torch_output = m1(
        torch.tensor(inputs, dtype=torch.float32), torch.tensor(hx, dtype=torch.float32)
    )
    mge_output = m2(
        mge.tensor(inputs, dtype=np.float32), mge.tensor(hx, dtype=np.float32)
    )

    np.testing.assert_allclose(
        torch_output[0].detach().numpy(), mge_output.numpy(), atol=1e-5
    )


def test_LSTM_forward():

    inputs = np.random.randn(6, 3, 10)

    hx = np.random.randn(2, 3, 20)
    cx = np.random.randn(2, 3, 20)

    m1 = torch.nn.LSTM(10, 20, 2, batch_first=False)
    m2 = LSTM(10, 20, 2, batch_first=False)
    # m1.eval()
    # m2.eval()

    for m in m1.parameters():
        m.data.fill_(1)
    for m in m2.parameters():
        M.init.fill_(m, 1)

    torch_output = m1(torch.tensor(inputs, dtype=torch.float32))
    mge_output = m2(mge.tensor(inputs, dtype=np.float32))

    torch_param = (
        torch.tensor(hx, dtype=torch.float32),
        torch.tensor(cx, dtype=torch.float32),
    )
    mge_param = (mge.tensor(hx, dtype=np.float32), mge.tensor(cx, dtype=np.float32))
    torch_output = m1(torch.tensor(inputs, dtype=torch.float32), torch_param)
    mge_output = m2(mge.tensor(inputs, dtype=np.float32), mge_param)

    np.testing.assert_allclose(
        torch_output[0].detach().numpy(), mge_output.numpy(), atol=1e-5
    )
