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
    mean_x = np.mean(x_train, axis=0)
    x_train = x_train - mean_x
    x_test = x_test - mean_x
    
    std_x = np.std(x_train, axis=0)
    x_train = x_train / std_x
    x_test = x_test / std_x
    
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