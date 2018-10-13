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
        
    return x_separated, y_separated, ids_separated
