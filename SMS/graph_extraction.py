from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
import numpy as np
import dgl
from dgl import DGLGraph
import torch

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}'.format(x, allowable_set))
    return list(map(lambda s:x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s:x == s, allowable_set))


def get_atom_features(atom, stereo, feature, explicit_H = False):

    possible_atom = ['S', 'Ge', 'O', 'C', 'I', 'N', '*', 'F', 'Cl', 'Se', 'Br', 'Si']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(),possible_atom)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2])
    
    atom_features += [int(i) for i in list('{0:06b}'.format(feature))]

    if not explicit_H:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    
    try:
        atom_features += one_of_k_encoding_unk(stereo, ['R','S'])
        atom_features += [atom.HasProp('_ChiralityPossible')]
    except:
        atom_features += [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array(atom_features)


def get_bond_features(bond):

    bond_type = bond.GetBondType()
    bond_features = [bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
                     bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(), bond.IsInRing()]
    bond_features += one_of_k_encoding_unk(str(bond.GetStereo()),['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE'])

    return np.array(bond_features)


def get_graph_from_sms(molecular_SMILES):
    
    G = DGLGraph()
    molecule = Chem.MolFromSmiles(molecular_SMILES)
    feature = rdDesc.GetFeatureInvariants(molecule)
    stereo = Chem.FindMolChiralCenters(molecule)
    chiral_centers = [0] * molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]
    
    G.add_nodes(molecule.GetNumAtoms())
    node_features = []
    edge_features = []
    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i)
        atom_i_features = get_atom_features(atom_i, chiral_centers[i], feature[i])
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edges(i, j)
                bond_ij_features = get_bond_features(bond_ij)
                edge_features.append(bond_ij_features)

    G.ndata['x'] = torch.from_numpy(np.array(node_features))
    G.edata['w'] = torch.from_numpy(np.array(edge_features))
    
    return G


