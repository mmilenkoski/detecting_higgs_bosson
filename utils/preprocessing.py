# -*- coding: utf-8 -*-
"""Functions for preprocessing data and feature engineering"""
import numpy as np

# list of column names
data_columns = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_deltaeta_jet_jet",
               "DER_mass_jet_jet", "DER_prodeta_jet_jet", "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt",
               "DER_pt_ratio_lep_tau", "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt",
               "PRI_tau_eta", "PRI_tau_phi", "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met", "PRI_met_phi",
               "PRI_met_sumet", "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
               "PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi", "PRI_jet_all_pt"]

# dictionary where key is the number of observed particles (PRI_jet_num), and the value is list of non-missing column names
data_columns_splited = {
    0: ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet'],
 
    1: ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_all_pt'],
 
    2: ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt'],
 
    3: ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']}

def get_idx_columns_for_PRI_jet_num():
    """Helper function for `split_data_in_four`.
    
    After splitting the data in subsets by the value of `PRI_jet_num` some of the columns have only nan values.
    This function returns the column indexes of the valid columns for every subset.
    
    Returns
    -------
    columns_for_PRI_jet_num: dict
        Dictionary containing the indexes of the valid columns.
    """
    columns_for_PRI_jet_num = {}
    for i in range(4):
        columns_for_PRI_jet_num[i] = [x in data_columns_splited[i] for x in data_columns]
    return columns_for_PRI_jet_num

def data_columns_get_index(column_name):
    """For a given column_name returns the corresponding column index in the original data matrix.
    
    Parameters
    ----------
    column_name: string
        String representing the name of the column.
    
    Returns
    -------
    index: int
        Integer representing the column index in the original data matrix.
    """
    index = data_columns.index(column_name)
    return index

def split_data_in_four(x, y, ids):
    """Splits the data by the value of PRI_jet_num.
    
    Splits the data by the value of the PRI_jet_num (0, 1, 2, 3). Returns dictionaries where the keys are the value of
    PRI_jet_num, and the values are the corresponding data matrix/labels/ids/indices. Removes columns with missing values from 
    each subset.
    
    Parameters
    ----------
    x: ndarray
        2D array representing the data. 
    y: ndarray
        1D array representing the labels of the data.
    ids: ndarray
        1D array representing the ids of the data.
    
    Returns
    -------
    x_separated: dict
        Dictionary of 2D arrays containing the splitted data.
    y_separated: dict
        Dictionary of 1D arrays containing the splitted labels.
    ids_separated: dict
        Dictionary of 1D arrays containing the splitted ids.
    indx: dict
        Dictionary of 1D arrays containing the original indices of the splited data in the original data matrix.
    """
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
    """Helper function used by `split_data_in_eight`.
    
    Splits the data in two subsets based on the value of the DER_mass_MMC, the first one contains real values for the mass
    and the second has "nan" values for mass.
    
    Parameters
    ----------
    x: ndarray
        2D array representing the data. 
    y: ndarray
        1D array representing the labels of the data.
    ids: ndarray
        1D array representing the ids of the data.
    indx: ndarray
        1D array representing the original indices of the data.
    
    Returns
    -------
    x_separated: dict
        Dictionary of 2D arrays containing the splitted data.
    y_separated: dict
        Dictionary of 1D arrays containing the splitted labels.
    ids_separated: dict
        Dictionary of 1D arrays containing the splitted ids.
    indx_separated: dict
        Dictionary of 1D arrays containing the original indices of the splited data in the original data matrix.
    """
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

def split_data_in_eight(x, y, ids):
    """Splits the data by the value of PRI_jet_num and by the DER_mass_MMC.
    
    After spliting the data by the PRI_jet_num it can also be splitted by the value of DER_mass_MMC into two subsets where the
    first one contains real values for the mass and the second subset has "nan" mass. Removes the mass column with missing
    values.
    
    Parameters
    ----------
    x: ndarray
        2D array representing the data. 
    y: ndarray
        1D array representing the labels of the data.
    ids: ndarray
        1D array representing the ids of the data.
    
    Returns
    -------
    x_separated: dict
        Dictionary of 2D arrays containing the splitted data.
    y_separated: dict
        Dictionary of 1D arrays containing the splitted labels.
    ids_separated: dict
        Dictionary of 1D arrays containing the splitted ids.
    indx_separated: dict
        Dictionary of 1D arrays containing the original indices of the splited data in the original data matrix.
    """
    x_separated = {}
    y_separated = {}
    ids_separated = {}
    indx_separated = {}
    t = 0
    
    x, y, ids, indx = split_data_in_four(x, y, ids)
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

def standardize(x_train, x_test):
    """Standardizes the test and train data.
    
    Standardizes each column of the test and train data by substracting from them the mean value of that column in the 
    training data and dividing each column with the standard deviation of that column in the traning data.
    
    Parameters
    ----------
    x_train: ndarray
        2D array representing the train feature matrix. 
    x_test: ndarray
        2D array representing the test feature matrix.
    
    Returns
    -------
    x_train: ndarray
        2D array representing the standardized train feature matrix.
    x_test: ndarray
        2D array representing the standardized test feature matrix.
    """
    assert x_train.shape[1] == x_test.shape[1]
    mean_x = np.mean(x_train, axis=0)
    x_train = x_train - mean_x
    x_test = x_test - mean_x
    
    std_x = np.std(x_train, axis=0)
    std_x[std_x == 0] = 1
    x_train = x_train / std_x
    x_test = x_test / std_x
    
    return x_train, x_test

def min_max_normalization(x_train, x_test):
    """Min_max standardization of the test and train data.
    
    Standardizes the values of each column of the test and train data to lie between zero and one, i.e. the minimum value
    is scaled to zero, the maximum value is scaled to one and the other values lie in between.
    
    Parameters
    ----------
    x_train: ndarray
        2D array representing the train feature matrix. 
    x_test: ndarray
        2D array representing the test feature matrix.
    
    Returns
    -------
    x_train: ndarray
        2D array representing the min_max standardized train feature matrix.
    x_test: ndarray
        2D array representing the min_max standardized test feature matrix.
    """
    assert x_train.shape[1] == x_test.shape[1]
    min_x = np.min(x_train, axis=0)
    max_x = np.max(x_train, axis=0)
    denum = max_x - min_x
    denum[denum == 0] = 1
    
    x_train = x_train - min_x
    x_train = x_train / denum
    
    x_test = x_test - min_x
    x_test = x_test / denum

    return x_train, x_test
        
def nan_to_mean(x_train, x_test):
    """Replaces the -999 (nan values) in the train and test data with the mean of each column in the train data.
    
    Parameters
    ----------
    x_train: ndarray
        2D array representing the train feature matrix. 
    x_test: ndarray
        2D array representing the test feature matrix.
    
    Returns
    -------
    x_train: ndarray
        2D array representing the train feature matrix cleaned from the -999 (nan) values.
    x_test: ndarray
        2D array representing the test feature matrix cleaned from the -999 (nan) values.
    """
    x_train[np.where(x_train == -999)] = np.nan
    means = np.nanmean(x_train, axis=0)
    inds = np.where(np.isnan(x_train))
    x_train[inds] = np.take(means, inds[1])
    
    x_test[np.where(x_test == -999)] = np.nan
    inds = np.where(np.isnan(x_test))
    x_test[inds] = np.take(means, inds[1])
    
    return x_train, x_test

def nan_to_median(x_train, x_test): 
    """Replaces the -999 (nan values) in the train and test data with the median of each column in the train data.
    
    Parameters
    ----------
    x_train: ndarray
        2D array representing the train feature matrix. 
    x_test: ndarray
        2D array representing the test feature matrix.
    
    Returns
    -------
    x_train: ndarray
        2D array representing the train feature matrix cleaned from the -999 (nan) values.
    x_test: ndarray
        2D array representing the test feature matrix cleaned from the -999 (nan) values.
    """
    x_train[np.where(x_train == -999)] = np.nan
    medians = np.nanmedian(x_train, axis=0)
    inds = np.where(np.isnan(x_train))
    x_train[inds] = np.take(medians, inds[1])
    
    x_test[np.where(x_test == -999)] = np.nan
    inds = np.where(np.isnan(x_test))
    x_test[inds] = np.take(medians, inds[1])
    
    return x_train, x_test

def build_poly(x, degree):
    """Polynomial expansion of x.
    
    Extends the feature matrix x by adding all polynomials of the features with degree less than or equal to the
    given degree parameter. Example, if we have a input [x, y] and degree 3 it generates [1, x, x^2, x^3, y, y^2, y^3].
    
    Parameters
    ----------
    x: ndarray
        2D array representing the feature matrix. 
    degree: int
        The maximum degree of the generated polynomial features.
     
    Returns
    -------
    poly: ndarray
        2D array representing the extended matrix.
    """
    if degree == -1:
        return x
    poly = np.ones((len(x), 1))
    for column in range(x.shape[1]):
        for deg in range(1, degree+1):
            poly = np.c_[poly, np.power(x[:, column], deg)]
    return poly

def get_angles_for_jet(i):
    """ Returns indices of columns with angle features for the model `i`.
    
    The models numbers have the following meaning:
    0 - jet 0 with mass
    1 - jet 0 without mass
    2 - jet 1 with mass
    3 - jet 1 without mass
    4 - jet 2 with mass
    5 - jet 2 without mass
    6 - jet 3 with mass
    7 - jet 3 without mass
    
    Parameters
    ----------
    i: int
        Model number.
    
    Returns
    -------
    angles: list
        Indices of columns with angle features for the corresponding model.
    """
    split_0 = [9, 10, 11, 12, 13, 14]
    split_1 = [9, 10, 11, 12, 13, 14, 18, 19, 20]
    split_2 = [13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26 ,27]
    
    if i in [0, 1]:
        angles = split_0
    if i in [2, 3]:
        angles = split_1
    if i in [4, 5, 6, 7]:
        angles = split_2
    
    if i % 2 == 0:
        return angles
    else:
        angles = [x-1 for x in angles]
        return angles
    
def get_phi_angles_for_jet(i):
    """ Returns indices of columns with phi-angle features for the model `i`.
    
    The models numbers have the following meaning:
    0 - jet 0 with mass
    1 - jet 0 without mass
    2 - jet 1 with mass
    3 - jet 1 without mass
    4 - jet 2 with mass
    5 - jet 2 without mass
    6 - jet 3 with mass
    7 - jet 3 without mass
    
    Parameters
    ----------
    i: int
        Model number.
    
    Returns
    -------
    angles: list
        Indices of columns with phi-angle features for the corresponding model.
    """
    split_0 = [11, 14, 16]
    split_1 = [11, 14, 16, 20]
    split_2 = [15, 18, 20, 24, 27]
    
    if i in [0, 1]:
        angels = split_0
    if i in [2, 3]:
        angels = split_1
    if i in [4, 5, 6, 7]:
        angels = split_2
    
    if i % 2 == 0:
        return angels
    else:
        return [x-1 for x in angels]


def adjust_cartesian_features(x_separated):
    """ Substracts reference angle from all phi-angles and transforms them to their sine and cosine components.
    
    Chooses `PRI_tau_phi` as reference angle, substracts it from all other phi-angles. Then, transforms each phi-angle to its
    sine and cosine component. Finally, removes original phi-angle features

    Parameters
    ----------
    x_separated: dict
        Dictionary of 2D arrays containing the splitted data before transformation.

    Returns
    -------
    x_separated: dict
        Dictionary of 2D arrays containing the transformed data.
    """
    for model in range(8):
        # phi-angles for current model
        phi_idx = get_phi_angles_for_jet(model)
        
        # reference angle
        tau_phi = x_separated[model][:, phi_idx[0]]
        
        for idx in phi_idx[1:]:
            # phi-angles to be transformed
            other_phi = x_separated[model][:, idx]
            # transform to sine and cosine components
            x_separated[model] = np.c_[x_separated[model], np.sin(other_phi - tau_phi)]
            x_separated[model] = np.c_[x_separated[model], np.cos(other_phi - tau_phi)]
        
        # delete original phi angles
        x_separated[model] = np.delete(x_separated[model], phi_idx, axis=1)
            
    return x_separated