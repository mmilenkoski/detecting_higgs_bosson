# -*- coding: utf-8 -*-
"""plot functions for project 1."""
import matplotlib.pyplot as plt


def cross_validation_visualization(lambds, mse_tr, mse_te, save_fig=None):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.show()
    if save_fig != None:
        plt.savefig(save_fig)