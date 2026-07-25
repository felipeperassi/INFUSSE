"""
Module containing paths to the data, logs, scripts and checkpoints.

"""
import os

ADJACENCIES_DIR = os.environ.get('ADJACENCIES_DIR', 'C:/Users/Usuario/Documents/Github/Tesis/Structures/adjacencies_sparse/')
CHECKPOINTS_DIR = os.environ.get('CHECKPOINTS_DIR', 'C:/Users/Usuario/Documents/Github/Tesis/INFUSSE/checkpoints/')
CM_DIR = os.environ.get('CM_DIR', 'C:/Users/Usuario/Documents/Github/Tesis/Structures/distance_matrices_exp/')
#CM_DIR = os.environ.get('CM_DIR', 'C:/Users/Usuario/Documents/Github/Tesis/Structures/contact_maps/')
DATA_DIR = os.environ.get('DATA_DIR', 'C:/Users/Usuario/Documents/Github/Tesis/INFUSSE/data/')
STRUCTURE_DIR = os.environ.get('STRUCTURE_DIR', 'C:/Users/Usuario/Documents/Github/Tesis/Structures/chothia_gcn/:C:/Users/Usuario/Documents/Github/Tesis/Structures/chothia_gcn_mice/').split(':')
#WGNM_DIR = os.environ.get('CM_DIR', '/Users/kevinmicha/Documents/all_structures/adjacencies_sparse_WGNM/')