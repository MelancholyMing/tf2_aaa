import numpy as np


class simplenet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


def numerical_graddient_2d(f, x):
    if x.ndim == 1:
        return _numerical_graddient_1d(f, x)

    else:
        grad = np.zeros_like(x)

        for idx, x in enumerate(x):
            grad[idx] = _numerical_graddient_1d(f, x)

    return grad


def _numerical_graddient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        # fxh1 = f(x)
        fxh1 = f()
        print('fxh1:', fxh1)

        x[idx] = float(tmp_val) - h
        # fxh2 = f(x)
        fxh2 = f()
        print('fxh2:', fxh2)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
    return grad


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))


if __name__ == '__main__':
    net = simplenet()
    print(net.W)
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    y = np.dot(x, net.W)
    print('y:', y)
    y_sm = softmax(y)
    print(y_sm)
    loss = cross_entropy_error(y_sm, t)
    print(loss)


    def f():
        print("x:", x)
        print("t:", t)
        return net.loss(x, t)


    # f = lambda _: net.loss(x, t)
    dw = numerical_graddient_2d(f, net.W)
    print(dw)
