from collections import defaultdict
import itertools
import numpy as np
import pandas as pd
import re
import torch
from scipy.spatial.distance import cdist
from dataclasses import dataclass

from infusse.config import STRUCTURE_DIR
from transformers import BertTokenizer, RoFormerTokenizer

def antibody_sequence_identity(seq1, seq2):
    r"""Computes the percentage of sequence identity.

    Parameters
    ----------
    seq1: list
        First sequence.
    seq2: list
        Second sequence.
    
    """
    if len(seq1) != len(seq2):
        return 0

    valid_aa = [(a, b) for a, b in zip(seq1, seq2) if a > 4 and b > 4]
    if not valid_aa:
        return 0
        
    matches = sum(1 for a, b in valid_aa if a == b)
    
    return matches / len(valid_aa)

def bootstrap_test(delta_graph, labels, ind_class='secondary_ab', B=100000, statistic='mean', compare_to_zero=False, all_pairwise=False):         
    """
    Performs pairwise bootstrap hypothesis tests to compare the means (or IQR) of multiple classes.

    Parameters
    ----------
    delta_graph: list of lists
        Observed values for each amino acid residue across complexes.
    labels: list of lists
        Labels for each residue, same shape as delta_graph.
    ind_class: str
        Attribute for which the mean of delta_graph is meant to be tested.
    B: int
        Number of bootstrap resamples (default 100000).
    statistic: str
        Statistic involved in the test (e.g., 'mean', 'iqr')
    compare_to_zero: bool
        If 'True' run comparison with zero mean.
    all_pairwise: bool
        If 'True' compare every unordered pair of classes.

    """
    
    if statistic not in ('mean', 'iqr'):
        raise ValueError("statistic must be 'mean' or 'iqr'")

    def get_statistic(arr):
        return (np.mean(arr) if statistic == 'mean' else np.subtract(*np.percentile(arr, [75, 25])))

    stat_label = 'means' if statistic == 'mean' else 'IQRs'

    delta_graph_flat = []
    labels_flat = []

    for delta_graph_sublist, label_sublist in zip(delta_graph, labels):
        if len(delta_graph_sublist) == len(label_sublist):
            delta_graph_flat.extend(delta_graph_sublist)
            labels_flat.extend(label_sublist)
    delta_graph_flat = np.array(delta_graph_flat)
    labels_flat = np.array(labels_flat)

    if ind_class == 'cdr_status':
        labels_flat = [1 if sec in [3, 4, 5] else 0 for sec in labels_flat]  
        label_names = ['FR', 'CDR']
    elif ind_class == 'entropy':
        ent_bits = np.array(labels_flat, dtype=float) / np.log(2)
        ds_bin = pd.qcut(ent_bits, q=2, labels=[0, 1])
        labels_flat = ds_bin.to_numpy(dtype=int, na_value=-1)
        label_names = ['Low', 'High']
    elif ind_class == 'secondary_ab':
        label_names = ['Helix (FR)', 'Strand (FR)', 'Loop (FR)', 'Helix (CDR)', 'Strand (CDR)', 'Loop (CDR)']
    elif ind_class == 'secondary_ag':
        label_names = ['Helix', 'Strand', 'Loop']
    elif ind_class == 'paratope':
        label_names = ['Non-paratope', 'Paratope']
    elif ind_class == 'epitope':
        label_names = ['Non-epitope', 'Epitope']

    unique_labels = np.unique(labels_flat)
    if ind_class in ['epitope', 'paratope']:
        unique_labels = np.array([0, 1])

    if compare_to_zero:
        for label in unique_labels:
            sample = delta_graph_flat[labels_flat == label]
            N      = len(sample)
            stat_obs = get_statistic(sample)         

            # We build the null distribution with mean zero, centre sample, resample, compute statistic...
            centred = sample - stat_obs
            t_b = []
            for _ in range(B):
                boot = np.random.choice(centred, size=N, replace=True)
                t_b.append(get_statistic(boot))

            t_b   = np.asarray(t_b)
            p_val = (t_b >= stat_obs).mean() if stat_obs >= 0 else (t_b <= stat_obs).mean()

            if p_val:
                print(f'{stat_label[:-1].capitalize()} for {label_names[label]} vs 0: '
                      f'{stat_obs} (p-value = {p_val}).')
            else:
                print(f'{stat_label[:-1].capitalize()} for {label_names[label]} vs 0: '
                      f'{stat_obs} (p-value < {1/B}).')
        return # stop here

    means = []
    for label in unique_labels:
        means.append((label, get_statistic(delta_graph_flat[labels_flat == label])))
    means.sort(key=lambda x: x[1], reverse=True)
    sorted_labels = [x[0] for x in means]

    if all_pairwise:
        pair_list = itertools.combinations(sorted_labels, 2)
    else:
        pair_list = zip(sorted_labels[:-1], sorted_labels[1:])  # original

    for label_high, label_low in pair_list:
        delta_high = delta_graph_flat[labels_flat == label_high]
        delta_low  = delta_graph_flat[labels_flat == label_low]

        N_high, N_low = len(delta_high), len(delta_low)
        mu_high = get_statistic(delta_high)
        mu_low  = get_statistic(delta_low)
        t_obs   = mu_high - mu_low

        t_b = []
        for _ in range(B):
            boot_high = np.random.choice(delta_graph_flat, N_high, True)
            boot_low  = np.random.choice(delta_graph_flat, N_low,  True)
            t_b.append(get_statistic(boot_high) - get_statistic(boot_low))

        p_value = (np.array(t_b) >= t_obs).mean()

        if p_value:
            print(f'Difference of {stat_label} between '
                  f'{label_names[label_high]} and {label_names[label_low]}: '
                  f'{t_obs} (p-value = {p_value}).')
        else:
            print(f'Difference of {stat_label} between '
                  f'{label_names[label_high]} and {label_names[label_low]}: '
                  f'{t_obs} (p-value < {1/B}).')
            
def compute_average_b_factors(b_amino_acids, b_factor_thr=100):
    unp = False
    if any(b_fact_element > b_factor_thr or b_fact_element < 0 for b_fact_element in b_amino_acids):
        unp = True
    averages = b_amino_acids
    mean_ = np.mean(averages)
    std_dev_ = np.std(averages)
    if std_dev_ == 0:
        raise Exception('All the atoms have the same B factor') 

    return (np.array(averages) - mean_) / std_dev_, unp

def count_consecutive_secondary(lst, cdr_status, delta_graph_flattened, M, valid_values, processed_indices):
    counts = {'FR': 0, 'CDR': 0, 'Total': 0}
    delta_graph_sums = {'FR': 0, 'CDR': 0, 'Total': 0}
    delta_graph_counts = {'FR': 0, 'CDR': 0, 'Total': 0}

    i = 0
    while i <= len(lst) - M:
        segment = lst[i:i+M]
        if all(val in valid_values for val in segment) and all(idx not in processed_indices for idx in range(i, i+M)):
            region_type = 'FR' if cdr_status[i] == 0 else 'CDR'
            counts[region_type] += 1
            counts['Total'] += 1

            # Add corresponding delta_graph values
            delta_graph_segment = delta_graph_flattened[i:i+M]
            delta_graph_sums[region_type] += sum(delta_graph_segment)
            delta_graph_sums['Total'] += sum(delta_graph_segment)
            delta_graph_counts[region_type] += M
            delta_graph_counts['Total'] += M

            # Mark indices as processed
            processed_indices.update(range(i, i+M))
            i += M  # skip this segment
        else:
            i += 1  # increment and check the next segment

    # Average delta_graph 
    avg_delta_graph = {
        'FR': delta_graph_sums['FR'] / delta_graph_counts['FR'] if delta_graph_counts['FR'] > 0 else 0,
        'CDR': delta_graph_sums['CDR'] / delta_graph_counts['CDR'] if delta_graph_counts['CDR'] > 0 else 0,
        'Total': delta_graph_sums['Total'] / delta_graph_counts['Total'] if delta_graph_counts['Total'] > 0 else 0,
    }

    return counts, avg_delta_graph

def encode_line(line, amino_acids, secondary_structure):
    '''
    Generating one-hot encoded entry
    '''

    aa_mapping = {aa: i for i, aa in enumerate(amino_acids)}
    ss_mapping = {ss: i + len(amino_acids) for i, ss in enumerate(secondary_structure)}

    # From STRIDES output
    aa_pos = 1
    ss_pos = 6

    split_line = line.split()
    aa = split_line[aa_pos] 
    #ss = split_line[ss_pos]

    #one_hot = np.zeros(len(amino_acids)+len(secondary_structure))
    one_hot = np.zeros(len(amino_acids))
    if aa in aa_mapping:
        one_hot[aa_mapping[aa]] = 1
    #if ss in ss_mapping:
    #    one_hot[ss_mapping[ss]] = 1

    #one_hot[-3] = float(split_line[7]) / 360
    #one_hot[-2] = float(split_line[8]) / 180
    #one_hot[-1] = float(split_line[9]) 
    
    return one_hot

def extract_list_of_residues(file_path):
    pdb = file_path[-8:-4]
    ca_index = 0
    h_res_list = []
    l_res_list = []
    h_idx_list = []
    l_idx_list = []

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.find('HCHAIN') != -1 or line.find('LCHAIN') != -1:
                h_chain = line[line.find('HCHAIN')+len('HCHAIN')+1]
                l_chain = line[line.find('LCHAIN')+len('LCHAIN')+1]

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            fields = re.split(r'\s+', line.strip())
            if fields[2] == 'CA':
                if line.startswith('ATOM') and (h_chain == line[slice(21, 22)] or (line[slice(21, 22)].upper() == h_chain and h_chain != l_chain)):# and int(line[slice(23, 26)]) <= 112:
                    h_res_list.append(fields[5])
                    h_idx_list.append(ca_index)
                    ca_index += 1
                elif line.startswith('ATOM') and ((line[slice(21, 22)].upper() == l_chain and h_chain != l_chain) or (line[slice(21, 22)] == l_chain.lower() and h_chain == l_chain)):# and int(line[slice(23, 26)]) <= 107:
                    l_res_list.append(fields[5])
                    l_idx_list.append(ca_index)
                    ca_index += 1

    return h_res_list, l_res_list, h_idx_list+l_idx_list

def find_cdr_positions(heavy_l, light_l):
    heavy_set = ['26', '32', '52', '56', '95', '102']
    light_set = ['24', '34', '50', '56', '89', '97']
    cdr_indices = []
    l = heavy_l + light_l

    def find_next(l, target):
        try:
            return l.index(target)
        except ValueError:
            return -1 
    
    for target in heavy_set:
        idx = find_next(l, target)
        if idx != -1:
            cdr_indices.append(idx)
    
    for target in light_set:
        idx = find_next(l[len(heavy_l):], target)
        if idx != -1:
            cdr_indices.append(len(heavy_l)+idx)
    
    return cdr_indices

def format_sequence(sequence):
    seq_with_spaces = ' '.join(sequence)
    final_sequence = seq_with_spaces.replace('-', '[PAD]')
    final_sequence = final_sequence.replace(':', '[SEP]')
    final_sequence = final_sequence.replace('?', '[UNK]')
    
    return final_sequence

def generate_one_hot_matrix(file_path):
    '''
    Reading and extracting valid lines
    '''

    amino_acids = [' ', 'ASP', 'GLY', 'SEC', 'LEU', 'ASN', 'THR', 'LYS', 'HIS', 'TYR', 'TRP', 'CYS', 'PRO', 'VAL', 'SER', 'PYL', 'ILE', 'GLU', 'PHE', 'XAA', 'GLN', 'ALA', 'ASX', 'GLX', 'ARG', 'MET']
    secondary_structure = ['310Helix', 'PiHelix', 'AlphaHelix', 'Bridge', 'Coil', 'Strand', 'Turn']

    with open(file_path, 'r') as file:
        lines = file.readlines()
        valid_lines = [line for line in lines if line.startswith('ASG')]
        #valid_lines = [line for line in valid_lines if int(''.join(filter(str.isdigit, line.split()[3])))<=107]
        #X = np.zeros((len(valid_lines), len(amino_acids)+len(secondary_structure)), dtype=np.float32)
        X = np.zeros((len(valid_lines), len(amino_acids)), dtype=np.float32)
        for i, line in enumerate(valid_lines):
            X[i] = encode_line(line, amino_acids, secondary_structure)
    return X

def generate_secondary(file_path):
    '''
    Reading and extracting secondary structure from STRIDES files
    '''
    ss_pos = 6
    secondary_structure = ['310Helix', 'PiHelix', 'AlphaHelix', 'Bridge', 'Coil', 'Strand', 'Turn']

    with open(file_path, 'r') as file:
        lines = file.readlines()
        valid_lines = [line for line in lines if line.startswith('ASG')]
        secondary = [secondary_structure.index(line.split()[ss_pos]) for line in valid_lines]

    return secondary

def get_epitope_members(data, errors_ag):
    if data:
        epitope = [1 if i in data['epitope'] else 0 for i in range(len(errors_ag))]
    else:
        # unbound
        epitope = []
    return epitope    

def get_first_digit(filename):
    for char in filename:
        if char.isdigit():
            return int(char)
    return -1 

def get_paratope_members(paratope_data, len_h, len_l):
    if paratope_data:
        heavy_paratope = [1 if i in paratope_data['heavy_paratope'] else 0 for i in range(len_h)]
        light_paratope = [1 if i in paratope_data['light_paratope'] else 0 for i in range(len_l)]
    else:
        # unbound
        heavy_paratope = [2 for i in range(len_h)]
        light_paratope = [2 for i in range(len_l)]
    return heavy_paratope + light_paratope

def get_tokenised_sequence(file_path, cssp=False):
    aa_pos = 1
    chain_pos = 2
    amino_acid_dictionary = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'SEC': 'U', 'PYL': 'O', 'XAA': 'X',
    ' ': ' ', 'UNK': '?',
    }
    h_chain_seq = ''
    l_chain_seq = ''
    ag_chain_seq = ''
    
    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.find('AGCHAIN') != -1 or line.find('HCHAIN') != -1 or line.find('LCHAIN') != -1:
                if line[line.find('AGCHAIN')+len('AGCHAIN')+1:line.find('AGCHAIN')+len('AGCHAIN')+5] != 'NONE':
                    ag_chain = line[line.find('AGCHAIN')+len('AGCHAIN')+1]
                    if line[line.find('AGCHAIN')+len('AGCHAIN')+2] == ';':
                        ag_chain_2 = line[line.find('AGCHAIN')+len('AGCHAIN')+3]
                        if line[line.find('AGCHAIN')+len('AGCHAIN')+4] == ';':
                            ag_chain_3 = line[line.find('AGCHAIN')+len('AGCHAIN')+5]
                        else: 
                            ag_chain_3 = None
                    else:
                        ag_chain_2 = None
                        ag_chain_3 = None
                else:
                    ag_chain = None
                    ag_chain_2 = None
                    ag_chain_3 = None
                h_chain = line[line.find('HCHAIN')+len('HCHAIN')+1]
                l_chain = line[line.find('LCHAIN')+len('LCHAIN')+1]
    if h_chain == l_chain == ag_chain:
        ag_chain = None
        
    if cssp:
        tokeniser_ab = RoFormerTokenizer.from_pretrained('alchemab/antiberta2-cssp')
    else:
        tokeniser_ab = RoFormerTokenizer.from_pretrained('alchemab/antiberta2')
    tokeniser_ag = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM'):
                atom_type = line[12:16].strip()
                if atom_type == 'CA':
                    chain_id = line[21] 
                    residue_name = line[17:20].strip()  
                    residue_number = line[22:26].strip() 

                    if residue_name in amino_acid_dictionary:
                        amino_acid = amino_acid_dictionary[residue_name]
                        
                        if (chain_id == h_chain or (chain_id.upper() == h_chain and h_chain != l_chain and h_chain != ag_chain)):# and int(line[slice(23, 26)]) <= 112:
                            h_chain_seq += amino_acid
                        elif ((chain_id.upper() == l_chain and h_chain != l_chain) or (chain_id == l_chain.lower() and h_chain == l_chain)):# and int(line[slice(23, 26)]) <= 107:
                            l_chain_seq += amino_acid
                        elif (chain_id.upper() in [ag_chain, ag_chain_2, ag_chain_3] and l_chain not in [ag_chain, ag_chain_2, ag_chain_3] and h_chain not in [ag_chain, ag_chain_2, ag_chain_3]) or (chain_id in [ag_chain.lower() if ag_chain is not None else None, ag_chain_2.lower() if ag_chain_2 is not None else None, ag_chain_3.lower() if ag_chain_3 is not None else None] and (l_chain in [ag_chain, ag_chain_2, ag_chain_3] or h_chain in [ag_chain, ag_chain_2, ag_chain_3])):
                            ag_chain_seq += amino_acid
    X_ab = f'{h_chain_seq}:{l_chain_seq}'
    C = [0] * len(h_chain_seq)
    C.extend([1] * len(l_chain_seq))

    if ag_chain_seq != '':
        X_ag = f'{X_ab}:{ag_chain_seq}'
        C.extend([2] * len(ag_chain_seq))
        #if '?' in X_ag:
        #    print(pdb_file)
        input_seq_ag = format_sequence(X_ag)
        inputs_ag = tokeniser_ag(input_seq_ag, return_tensors='pt')['input_ids'][0]
        #inputs = torch.cat((inputs, torch.Tensor([1]), inputs_ag))
        inputs = inputs_ag #torch.cat((inputs, inputs_ag))
    else:
        input_seq_ab = format_sequence(X_ab)
        inputs = tokeniser_ag(input_seq_ab, return_tensors='pt')['input_ids'][0]
    return inputs, C, X_ab

def get_antigen_only(lists, heavy, light):
    lists_ag = [lists[i][heavy[i]+light[i]:] for i in range(len(lists))]
    
    return lists_ag

def get_variable_region_only(lists, heavy, light, heavy_v, light_v):
    lists_v = [list(np.concatenate((lists[i][:heavy_v[i]], lists[i][heavy[i]:heavy[i] + light_v[i]]))) for i in range(len(lists))]
    
    return lists_v

def is_in_cdr(position, chain_type):
    # Heavy chain CDR ranges
    heavy_cdr_ranges = [
        range(26, 33),  # CDR1: 26-32 inclusive
        range(52, 57),  # CDR2: 52-56 inclusive
        range(95, 103)  # CDR3: 95-102 inclusive
    ]
    # Light chain CDR ranges
    light_cdr_ranges = [
        range(24, 35),  # CDR1: 24-34 inclusive
        range(50, 57),  # CDR2: 50-56 inclusive
        range(89, 98)   # CDR3: 89-97 inclusive
    ]
    # Parse position into numeric part and optional letter suffix (e.g., "100A")
    numeric_part = int(''.join(filter(str.isdigit, position)))
    
    # Check ranges based on chain type
    if chain_type == 'H':
        return any(numeric_part in cdr_range for cdr_range in heavy_cdr_ranges)
    elif chain_type == 'L':
        return any(numeric_part in cdr_range for cdr_range in light_cdr_ranges)
    return False

def parse_pdb(file_path):
    b_factors_heavy_chain = []
    b_factors_light_chain = []
    b_factors_antigens = []

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.find('AGCHAIN') != -1 or line.find('HCHAIN') != -1 or line.find('LCHAIN') != -1:
                if line[line.find('AGCHAIN')+len('AGCHAIN')+1:line.find('AGCHAIN')+len('AGCHAIN')+5] != 'NONE':
                    ag_chain = line[line.find('AGCHAIN')+len('AGCHAIN')+1]
                    if line[line.find('AGCHAIN')+len('AGCHAIN')+2] == ';':
                        ag_chain_2 = line[line.find('AGCHAIN')+len('AGCHAIN')+3]
                        if line[line.find('AGCHAIN')+len('AGCHAIN')+4] == ';':
                            ag_chain_3 = line[line.find('AGCHAIN')+len('AGCHAIN')+5]
                        else: 
                            ag_chain_3 = None
                    else:
                        ag_chain_2 = None
                        ag_chain_3 = None
                else:
                    ag_chain = None
                    ag_chain_2 = None
                    ag_chain_3 = None
                h_chain = line[line.find('HCHAIN')+len('HCHAIN')+1]
                l_chain = line[line.find('LCHAIN')+len('LCHAIN')+1]

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM'):
                chain_id = line[slice(21, 22)].strip()  
                fields = re.split(r'\s+', line.strip())
                
                if fields[2] == 'CA': 
                    fields[-2] = '.'.join(fields[-2].split('.')[-2:]) if fields[-2].count('.') == 2 else fields[-2]
                    b_factor = float(fields[-2])
                    # Process heavy chain first
                    if (chain_id == h_chain or (chain_id.upper() == h_chain and h_chain != l_chain and h_chain != ag_chain)):# and int(line[slice(23, 26)]) <= 112:
                        b_factors_heavy_chain.append(b_factor)
                    # Then light chain 
                    elif ((l_chain == chain_id.upper() and h_chain != l_chain) or (l_chain.lower() == chain_id and h_chain == l_chain)):# and int(line[slice(23, 26)]) <= 107:
                        b_factors_light_chain.append(b_factor)
                    # Finally antigen chains
                    elif (chain_id.upper() in [ag_chain, ag_chain_2, ag_chain_3] and l_chain not in [ag_chain, ag_chain_2, ag_chain_3] and h_chain not in [ag_chain, ag_chain_2, ag_chain_3]) or (chain_id in [ag_chain.lower() if ag_chain is not None else None, ag_chain_2.lower() if ag_chain_2 is not None else None, ag_chain_3.lower() if ag_chain_3 is not None else None] and (l_chain in [ag_chain, ag_chain_2, ag_chain_3] or h_chain in [ag_chain, ag_chain_2, ag_chain_3])):
                        b_factors_antigens.append(b_factor)

            elif line.startswith('ENDMDL'):
                break
                
    return b_factors_heavy_chain + b_factors_light_chain + b_factors_antigens

# Residue class to store chain, seq number, CA location and heavy atoms locations
@dataclass
class Residue:
    chain_id: str
    seq_number: str
    ca_loc : tuple[float, float, float]
    heavy_atoms_loc: list[tuple[float, float, float]]

def chains_from_pdb(file_path): # Returns lists of Residue objects for hchain, lchain & ags heavy atoms
    amino_acid_dictionary = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'SEC': 'U', 'PYL': 'O', 'XAA': 'X',
    ' ': ' ', 'UNK': '?',
    }
    # Header parsing to identify chains
    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.find('AGCHAIN') != -1 or line.find('HCHAIN') != -1 or line.find('LCHAIN') != -1:
                if line[line.find('AGCHAIN')+len('AGCHAIN')+1:line.find('AGCHAIN')+len('AGCHAIN')+5] != 'NONE':
                    ag_chain = line[line.find('AGCHAIN')+len('AGCHAIN')+1]
                    if line[line.find('AGCHAIN')+len('AGCHAIN')+2] == ';':
                        ag_chain_2 = line[line.find('AGCHAIN')+len('AGCHAIN')+3]
                        if line[line.find('AGCHAIN')+len('AGCHAIN')+4] == ';':
                            ag_chain_3 = line[line.find('AGCHAIN')+len('AGCHAIN')+5]
                        else: 
                            ag_chain_3 = None
                    else:
                        ag_chain_2 = None
                        ag_chain_3 = None
                else:
                    ag_chain = None
                    ag_chain_2 = None
                    ag_chain_3 = None
                h_chain = line[line.find('HCHAIN')+len('HCHAIN')+1]
                l_chain = line[line.find('LCHAIN')+len('LCHAIN')+1]

    # Lists to store residues of each chain
    h_chain_heavy_atoms = []
    l_chain_heavy_atoms = []
    ag_heavy_atoms = []

    # Init vars
    current_res = None
    current_id = None
    current_seq_number = None
    prev_dest = None

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                chain_id = line[slice(21, 22)].strip()
                seq_number = line[slice(22, 27)].strip()
                residue_name = line[17:20].strip()
                x, y, z = float(line[slice(30, 38)].strip()), float(line[slice(38, 46)].strip()), float(line[slice(46, 54)].strip())

                if not atom_name.startswith('H'): # Only heavy atoms
                    # ag case
                    if (chain_id.upper() in [ag_chain, ag_chain_2, ag_chain_3] and l_chain not in [ag_chain, ag_chain_2, ag_chain_3] and h_chain not in [ag_chain, ag_chain_2, ag_chain_3]) or (chain_id in [ag_chain.lower() if ag_chain is not None else None, ag_chain_2.lower() if ag_chain_2 is not None else None, ag_chain_3.lower() if ag_chain_3 is not None else None] and (l_chain in [ag_chain, ag_chain_2, ag_chain_3] or h_chain in [ag_chain, ag_chain_2, ag_chain_3])):
                        dest = ag_heavy_atoms

                    # h_chain case
                    elif (chain_id == h_chain or (chain_id.upper() == h_chain and h_chain != l_chain and h_chain != ag_chain)):
                        dest = h_chain_heavy_atoms

                    # l_chain case  
                    elif ((l_chain == chain_id.upper() and h_chain != l_chain) or (l_chain.lower() == chain_id and h_chain == l_chain)):
                        dest = l_chain_heavy_atoms
                    
                    else: 
                        continue # Skip atoms that don't belong to any of the chains of interest (errors)
                    
                    if current_id is None or (chain_id, seq_number) != (current_id, current_seq_number): # We find a new res
                        if current_id is not None: # Not the first res, save the previous before starting the new one
                            if current_res.ca_loc is not None: # Only save residues with CA location (i.e., valid residues)
                                prev_dest.append(current_res)

                        current_res = Residue(chain_id=chain_id, seq_number=seq_number, ca_loc=None, heavy_atoms_loc=[])
                        current_id, current_seq_number = chain_id, seq_number
                    
                    if residue_name not in amino_acid_dictionary:
                        current_id = None 
                        continue
                    current_res.heavy_atoms_loc.append((x, y, z)) # Append the heavy atom to the current res
                    current_res.ca_loc = (x, y, z) if atom_name == 'CA' else current_res.ca_loc # Update CA location if it's a CA atom
                    prev_dest = dest

        # Save the last residue after the loop
        if current_res is not None and current_res.ca_loc is not None:
            prev_dest.append(current_res)

    return h_chain_heavy_atoms, l_chain_heavy_atoms, ag_heavy_atoms

def compute_epitope_paratope(h_chain, l_chain, ag, threshold=5.0, compute_paratope=False):     
    ab_matrix_full = np.array([atom_loc for res in h_chain + l_chain for atom_loc in res.heavy_atoms_loc]) # All heavy atoms of the ab (for epitope computation)
    ag_matrix_full = np.array([atom_loc for res in ag for atom_loc in res.heavy_atoms_loc]) # All heavy atoms of the ag (for paratope computation)

    # Epitope computation (1 for epitope res, 0 for rest)
    epitope_labels = []
    for res in ag:
        # if res.ca_loc is None:
        #     epitope_labels.append(0)
        #     continue
        ag_res_matrix = np.array(res.heavy_atoms_loc)
        dist_matrix = cdist(ag_res_matrix, ab_matrix_full) # Dist[h_atoms ag res, h_atoms of the full ab]
        epitope_labels.append(1 if np.any(dist_matrix <= threshold) else 0) 
    
    # Paratope computation (-1 for paratope res, 0 for rest) -> check the -1
    paratope_labels = []
    if compute_paratope:
        for res in h_chain + l_chain:
            if res.ca_loc is None:
                paratope_labels.append(0)
                continue
            ab_res_matrix = np.array(res.heavy_atoms_loc)
            dist_matrix = cdist(ab_res_matrix, ag_matrix_full) # Dist[h_atoms ab res, h_atoms of the full ag]
            paratope_labels.append(-1 if np.any(dist_matrix <= threshold) else 0) 

    else: # paratope labels not computed, set to 0 for all residues
        paratope_labels = [0] * (len(h_chain) + len(l_chain))

    return paratope_labels, epitope_labels

def preprocess_interpretability(errors, errors_seq, secondary, ds, heavy, light, paratope_epitope):
    secondary_v = secondary.copy()
    # Computing delta_graph
    heavy_v = [] # Only variable region
    light_v = [] # Only variable region
    paratope_m = []
    epitope_m = []
    delta_graph = [[es - e for e, es in zip(e_list, s_list)] for e_list, s_list in zip(errors, errors_seq)]
    delta_graph_ag = get_antigen_only(delta_graph, heavy, light)

    for i, ds_dict in enumerate(ds):
        len_h = len([k for k in ds_dict if k.startswith('H')]) # HC variable region
        len_l = len([k for k in ds_dict if k.startswith('L')]) # LC variable region
        heavy_v.append(len_h)
        light_v.append(len_l)
        
        paratope_m.append(get_paratope_members(paratope_epitope[i], len_h, len_l))
        epitope_m.append(get_epitope_members(paratope_epitope[i], delta_graph_ag[i]))
        ds[i] = list(ds_dict.values())
        secondary_v[i] = list(np.concatenate((secondary[i][:len_h], secondary[i][heavy[i]:heavy[i]+len_l])))

        # Secondary structure classes (variable region only)
        converted_secondary = []
        for key, sec_value in zip(ds_dict.keys(), secondary_v[i]):
            position = key[1:]  # Position (e.g., '26', '100A')
            chain_type = key[0]  # Chain type ('H', 'L')
            cdr_is = is_in_cdr(position, chain_type)
            if sec_value == 0 and cdr_is:
                converted_secondary.append(3)
            elif sec_value == 1 and cdr_is:
                converted_secondary.append(4)
            elif sec_value == 2 and cdr_is:
                converted_secondary.append(5)
            else:
                converted_secondary.append(sec_value)
        secondary_v[i] = converted_secondary

    return delta_graph, secondary_v, ds, heavy_v, light_v, epitope_m, paratope_m

def separate_tokenised_chains(tensor):
    for i in range(1, len(tensor)):
        if tensor[i] == 1:
            return tensor[:i], tensor[i+1:] 
    
    return tensor, None 

def sort_keys(keys):
    def res_id_sorting(key):
        match = re.match(r'(\d+)([A-Z]?)([HL])$', key)
        residue_number = int(match.group(1))
        residue_letter = match.group(2) or ''
        chain_letter = match.group(3)
        
        return (chain_letter == 'L', residue_number, residue_letter or '')

    return sorted(keys, key=res_id_sorting)