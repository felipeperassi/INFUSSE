"""
Module containing paths to the data, logs, scripts and checkpoints.

"""
import os

ADJACENCIES_DIR = os.environ.get('ADJACENCIES_DIR', 'C:/Users/Usuario/Documents/Github/Tesis/Structures/adjacencies_sparse/')
CHECKPOINTS_DIR = os.environ.get('CHECKPOINTS_DIR', 'C:/Users/Usuario/Documents/Github/Tesis/INFUSSE/checkpoints/')
CM_DIR = os.environ.get('CM_DIR', 'C:/Users/Usuario/Documents/Github/Tesis/Structures/distance_matrices_exp/')
DATA_DIR = os.environ.get('DATA_DIR', 'C:/Users/Usuario/Documents/Github/Tesis/INFUSSE/data/')
STRUCTURE_DIR = os.environ.get('STRUCTURE_DIR', 'C:/Users/Usuario/Documents/Github/Tesis/Structures/chothia_gcn/:C:/Users/Usuario/Documents/Github/Tesis/Structures/chothia_gcn_mice/').split(':')
WGNM_DIR = os.environ.get('WGNM_DIR', 'C:/Users/Usuario/Documents/Github/Tesis/Structures/distance_matrices_exp/')

DEFAULT_GRAPH = 'wgnm'
GRAPH_TYPES = ('wgnm', 'gnm', 'bagpype')
EDGE_DATA_FILES = {
    'wgnm': 'edge_data.pt',
    'gnm': 'edge_data_gnm.pt',
    'bagpype': 'edge_data_bagpype.pt',
}

VAL_SEED = 0
SEED = 0
