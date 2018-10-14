# -*- coding: utf-8 -*-
"""user defined function for preprocessing data."""
import numpy as np

data_columns = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_deltaeta_jet_jet",
               "DER_mass_jet_jet", "DER_prodeta_jet_jet", "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt",
               "DER_pt_ratio_lep_tau", "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt",
               "PRI_tau_eta", "PRI_tau_phi", "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met", "PRI_met_phi",
               "PRI_met_sumet", "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
               "PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi", "PRI_jet_all_pt"]

data_columns_splited = {0: ['DER_mass_MMC',
  'DER_mass_transverse_met_lep',
  'DER_mass_vis',
  'DER_pt_h',
  'DER_deltar_tau_lep',
  'DER_pt_tot',
  'DER_sum_pt',
  'DER_pt_ratio_lep_tau',
  'DER_met_phi_centrality',
  'PRI_tau_pt',
  'PRI_tau_eta',
  'PRI_tau_phi',
  'PRI_lep_pt',
  'PRI_lep_eta',
  'PRI_lep_phi',
  'PRI_met',
  'PRI_met_phi',
  'PRI_met_sumet'],
 1: ['DER_mass_MMC',
  'DER_mass_transverse_met_lep',
  'DER_mass_vis',
  'DER_pt_h',
  'DER_deltar_tau_lep',
  'DER_pt_tot',
  'DER_sum_pt',
  'DER_pt_ratio_lep_tau',
  'DER_met_phi_centrality',
  'PRI_tau_pt',
  'PRI_tau_eta',
  'PRI_tau_phi',
  'PRI_lep_pt',
  'PRI_lep_eta',
  'PRI_lep_phi',
  'PRI_met',
  'PRI_met_phi',
  'PRI_met_sumet',
  'PRI_jet_leading_pt',
  'PRI_jet_leading_eta',
  'PRI_jet_leading_phi',
  'PRI_jet_all_pt'],
 2: ['DER_mass_MMC',
  'DER_mass_transverse_met_lep',
  'DER_mass_vis',
  'DER_pt_h',
  'DER_deltaeta_jet_jet',
  'DER_mass_jet_jet',
  'DER_prodeta_jet_jet',
  'DER_deltar_tau_lep',
  'DER_pt_tot',
  'DER_sum_pt',
  'DER_pt_ratio_lep_tau',
  'DER_met_phi_centrality',
  'DER_lep_eta_centrality',
  'PRI_tau_pt',
  'PRI_tau_eta',
  'PRI_tau_phi',
  'PRI_lep_pt',
  'PRI_lep_eta',
  'PRI_lep_phi',
  'PRI_met',
  'PRI_met_phi',
  'PRI_met_sumet',
  'PRI_jet_leading_pt',
  'PRI_jet_leading_eta',
  'PRI_jet_leading_phi',
  'PRI_jet_subleading_pt',
  'PRI_jet_subleading_eta',
  'PRI_jet_subleading_phi',
  'PRI_jet_all_pt'],
 3: ['DER_mass_MMC',
  'DER_mass_transverse_met_lep',
  'DER_mass_vis',
  'DER_pt_h',
  'DER_deltaeta_jet_jet',
  'DER_mass_jet_jet',
  'DER_prodeta_jet_jet',
  'DER_deltar_tau_lep',
  'DER_pt_tot',
  'DER_sum_pt',
  'DER_pt_ratio_lep_tau',
  'DER_met_phi_centrality',
  'DER_lep_eta_centrality',
  'PRI_tau_pt',
  'PRI_tau_eta',
  'PRI_tau_phi',
  'PRI_lep_pt',
  'PRI_lep_eta',
  'PRI_lep_phi',
  'PRI_met',
  'PRI_met_phi',
  'PRI_met_sumet',
  'PRI_jet_leading_pt',
  'PRI_jet_leading_eta',
  'PRI_jet_leading_phi',
  'PRI_jet_subleading_pt',
  'PRI_jet_subleading_eta',
  'PRI_jet_subleading_phi',
  'PRI_jet_all_pt']}

def get_idx_columns_for_PRI_jet_num():
    columns_for_PRI_jet_num = {}
    for i in range(4):
        columns_for_PRI_jet_num[i] = [x in data_columns_splited[i] for x in data_columns]
    return columns_for_PRI_jet_num

def data_columns_get_index(item):
    return data_columns.index(item)

def split_data_by_PRI_jet_num(x, y, ids):
    columns = get_idx_columns_for_PRI_jet_num()
    pri_jet_num = x[:,data_columns_get_index("PRI_jet_num")]
    
    indx = {}
    for i in range(4):
        indx[i] = (pri_jet_num == i)
    
    x_separated = {}
    y_separated = {}
    ids_separated = {}
    for i in range(4):
        x_separated[i] = x[indx[i]]
        x_separated[i] = x_separated[i][:, columns[i]]
        y_separated[i] = y[indx[i]]
        ids_separated[i] = ids[indx[i]]
        
    return x_separated, y_separated, ids_separated, indx

def split_data_by_DER_mass_MMC_helper(x, y, ids, indx):
    null_ids = x[:, 0] == -999
    not_null_ids = ~null_ids
    
    x_separated = {}
    y_separated = {}
    ids_separated = {}
    indx_separated = {}
    
    x_separated[0] = x[not_null_ids]
    y_separated[0] = y[not_null_ids]
    ids_separated[0] = ids[not_null_ids]
    indx_separated[0] = indx[not_null_ids]
    
    x_separated[1] = x[null_ids, 1:]
    y_separated[1] = y[null_ids]
    ids_separated[1] = ids[null_ids]
    indx_separated[1] = indx[null_ids]

    return x_separated, y_separated, ids_separated, indx_separated

def split_data_by_DER_mass_MMC(x, y, ids):
    x_separated = {}
    y_separated = {}
    ids_separated = {}
    indx_separated = {}
    t = 0
    
    x, y, ids, indx = split_data_by_PRI_jet_num(x, y, ids)
    for i in range(4):
        indx[i], = np.where(indx[i])
        x_sep, y_sep, ids_sep, indx_sep = split_data_by_DER_mass_MMC_helper(x[i], y[i], ids[i], indx[i])
        for j in range(2):
            x_separated[t+j] = x_sep[j]
            y_separated[t+j] = y_sep[j]
            ids_separated[t+j] = ids_sep[j]
            indx_separated[t+j] = indx_sep[j]
        t = t + 2

    return x_separated, y_separated, ids_separated, indx_separated

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def standardize(x_train, x_test):
    """Standardize the original data set."""
    assert x_train.shape[1] == x_test.shape[1]
    mean_x = np.mean(x_train, axis=0)
    x_train = x_train - mean_x
    x_test = x_test - mean_x
    
    std_x = np.std(x_train, axis=0)
    x_train = x_train / std_x
    x_test = x_test / std_x
    
    return x_train, x_test

def min_max_normalization(x_train, x_test):
    """Standardize the original data set."""
    
    assert x_train.shape[1] == x_test.shape[1]
    min_x = np.min(x_train, axis=0)
    max_x = np.max(x_train, axis=0)
    denum = max_x - min_x
    
    x_train = x_train - min_x
    x_train = x_train / denum
    
    x_test = x_test - min_x
    x_test = x_test / denum

    return x_train, x_test

def build_k_indices(y, k_fold, seed, shuffle):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    if shuffle:
        indices = np.random.permutation(num_row)
    else:
        indices = num_row
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def k_fold_split(y, x, n_splits, seed=0, shuffle=True):
    """return train index and test index.
        sci-kit like function!
    """
    assert y.shape[0] == x.shape[0]
    k_indices = build_k_indices(y, n_splits, seed, shuffle)
    for n in range(n_splits):
        te_indice = k_indices[n]
        tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == n)]
        tr_indice = tr_indice.reshape(-1)
        yield tr_indice, te_indice
        
def nan_to_mean(x_train, x_test):
    """
    This method is used to replace -999 values with the mean of each column
    :param x: matrix X of training
    :param testx: matrix X of testing
    :return: the two matrix after substitution of each -999 value with the mean
    """
    x_train[np.where(x_train == -999)] = np.nan
    means = np.nanmean(x_train, axis=0)
    inds = np.where(np.isnan(x_train))
    x_train[inds] = np.take(means, inds[1])
    
    inds = np.where(np.isnan(x_test))
    x_test[inds] = np.take(means, inds[1])
    
    return x_train, x_test

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    if degree == -1:
        return x
    poly = np.ones((len(x), 1))
    for column in range(x.shape[1]):
        for deg in range(1, degree+1):
            poly = np.c_[poly, np.power(x[:, column], deg)]
    return poly

def index_of_PRI_tau_phi(model):
    if model == 0:
        return 11
    elif model == 1:
        return 10
    elif model == 2:
        return 11
    elif model == 3:
        return 10
    elif model == 4:
        return 15
    elif model == 5:
        return 14
    elif model == 6:
        return 15
    elif model == 7:
        return 14


def index_of_PRI_lep_phi(model):
    if model == 0:
        return 14
    elif model == 1:
        return 13
    elif model == 2:
        return 14
    elif model == 3:
        return 13
    elif model == 4:
        return 18
    elif model == 5:
        return 17
    elif model == 6:
        return 18
    elif model == 7:
        return 17


def index_of_PRI_met_phi(model):
    if model == 0:
        return 16
    elif model == 1:
        return 15
    elif model == 2:
        return 16
    elif model == 3:
        return 15
    elif model == 4:
        return 20
    elif model == 5:
        return 19
    elif model == 6:
        return 20
    elif model == 7:
        return 19

def index_of_PRI_jet_leading_phi(model):
    if model == 0:
        print('You called a non-existing parameter')
        return np.nan
    elif model == 1:
        print('You called a non-existing parameter')
        return np.nan
    elif model == 2:
        return 20
    elif model == 3:
        return 19
    elif model == 4:
        return 24
    elif model == 5:
        return 23
    elif model == 6:
        return 24
    elif model == 7:
        return 23


def index_of_PRI_jet_subleading_phi(model):
    if model == 0:
        print('You called a non-existing parameter')
        return np.nan
    elif model == 1:
        print('You called a non-existing parameter')
        return np.nan
    elif model == 2:
        print('You called a non-existing parameter')
        return np.nan
    elif model == 3:
        print('You called a non-existing parameter')
        return np.nan
    elif model == 4:
        return 27
    elif model == 5:
        return 26
    elif model == 6:
        return 27
    elif model == 7:
        return 26

