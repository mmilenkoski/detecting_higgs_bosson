# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np

def compute_loss(y, tx, w):
    n = len(y)
    e = y - np.dot(tx, w)
    loss = np.sum(1/(2*n) * np.power(e, 2))
    return loss
	
def compute_gradient(y, tx, w):
    N = len(y)
    e = y - np.dot(tx, w)
    gradient = -1/N * np.dot(np.transpose(tx), e)
    return gradient
	
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws



