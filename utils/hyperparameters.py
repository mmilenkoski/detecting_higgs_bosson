# -*- coding: utf-8 -*-

def get_lambda(model):
    best_lambas = [1000.0, 0.001, 100.0, 0.001, 100.0, 100.0, 0.001, 100.0]
    return best_lambas[model]

def get_gamma(model):
    best_gammas = [1e-6, 1e-6, 1e-6, 1e-5, 1e-6, 1e-05, 1e-05, 1e-05]
    return best_gammas[model]

def get_poly_degree():
    return 3

def get_max_iters():
    return 2000

def get_number_of_models():
    return 8