"""
Module containing paths to the data, logs, scripts and checkpoints.

"""
import os

ADJACENCIES_DIR = os.environ.get('ADJACENCIES_DIR', '/Users/kevinmicha/Documents/all_structures/adjacencies_sparse/')
CHECKPOINTS_DIR = os.environ.get('CHECKPOINTS_DIR', '../checkpoints/')
CM_DIR = os.environ.get('CM_DIR', '/Users/kevinmicha/Documents/all_structures/contact_maps/')
DATA_DIR = os.environ.get('DATA_DIR', '../data/')
STRUCTURE_DIR = os.environ.get('STRUCTURE_DIR', '/Users/kevinmicha/Documents/all_structures/chothia_gcn/:/Users/kevinmicha/Documents/all_structures/chothia_gcn_mice/').split(':')
WGNM_DIR = os.environ.get('WGNM_DIR', '/Users/kevinmicha/Documents/all_structures/distance_matrices_exp/')

DEFAULT_GRAPH = 'wgnm'
GRAPH_TYPES = ('wgnm', 'gnm', 'bagpype')
EDGE_DATA_FILES = {
    'wgnm': 'edge_data.pt',
    'gnm': 'edge_data_gnm.pt',
    'bagpype': 'edge_data_bagpype.pt',
}
