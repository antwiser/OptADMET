from .build_dataset import smiles2graphs_save
import torch
from .MY_GNN import EarlyStopping, MTRGCN, MTGAT, MTGCN
from dgl.data.graph_serialize import load_graphs
import dgl
import os
import time
import pandas as pd
import sys
from rdkit import Chem

import numpy as np

np.seterr(divide='ignore', invalid='ignore')

start = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from collections.abc import Iterable
from itertools import combinations
import numpy as np

from rdkit import RDConfig
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors, Lipinski, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

from .admetConfig import SmartDir
from .substructure_filter import SubstrutureFilter

from .medicinalchemistry import MedicinalChemistry
from math import pi, log10


def pred(args, bin_path, gnn_name):
    assert gnn_name in ['gcn', 'gat', 'rgcn'], \
        'Expect gnn name to be "gcn", "gat" or "rgcn",  got {}'.format(gnn_name)
    if gnn_name == 'rgcn':
        model = MTRGCN(in_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                       n_tasks=args['task_number'], rgcn_drop_out=args['rgcn_drop_out'],
                       classifier_hidden_feats=args['classifier_hidden_feats'], dropout=args['drop_out'],
                       loop=args['loop'])
        stopper = EarlyStopping(patience=args['patience'], task_name=args['model_name'], mode=args['mode'])
        model.to(args['device'])
        stopper.load_checkpoint(model)
        model.eval()
        with torch.no_grad():
            graphs, _ = load_graphs(bin_path)
            bg = dgl.batch(graphs)
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits = model(bg, atom_feats, bond_feats, norm=None)
            logits_c = logits[:, :args['classification_num']]
            logits_r = logits[:, args['classification_num']:]
            pred_c = torch.sigmoid(logits_c.detach().cpu())
            pred_r = logits_r.detach().cpu()
        pred_c_pd = pd.DataFrame(pred_c.numpy(), columns=args['classification_task'])
        pred_r_pd = pd.DataFrame(pred_r.numpy(), columns=args['regression_task'])
        pred_pd = pd.concat([pred_c_pd, pred_r_pd], axis=1)
        return pred_pd
    if gnn_name == 'gcn':
        model = MTGCN(in_feats=args['in_feats'], gcn_hidden_feats=args['gcn_hidden_feats'],
                      n_tasks=args['task_number'], gcn_drop_out=args['gcn_drop_out'],
                      classifier_hidden_feats=args['classifier_hidden_feats'], dropout=args['drop_out'])
        stopper = EarlyStopping(patience=args['patience'], task_name=args['model_name'], mode=args['mode'])
        model.to(args['device'])
        stopper.load_checkpoint(model)
        model.eval()
        with torch.no_grad():
            graphs, _ = load_graphs(bin_path)
            bg = dgl.batch(graphs)
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            logits = model(bg, atom_feats)
            logits_c = logits[:, :args['classification_num']]
            logits_r = logits[:, args['classification_num']:]
            pred_c = torch.sigmoid(logits_c.detach().cpu())
            pred_r = logits_r.detach().cpu()
        pred_c_pd = pd.DataFrame(pred_c.numpy(), columns=args['classification_task'])
        pred_r_pd = pd.DataFrame(pred_r.numpy(), columns=args['regression_task'])
        pred_pd = pd.concat([pred_c_pd, pred_r_pd], axis=1)
        return pred_pd
    if gnn_name == 'gat':
        model = MTGAT(in_feats=args['in_feats'], gat_hidden_feats=args['gat_hidden_feats'],
                      n_tasks=args['task_number'], gat_drop_out=args['gat_drop_out'], num_heads=args['num_heads'],
                      classifier_hidden_feats=args['classifier_hidden_feats'], dropout=args['drop_out'])
        stopper = EarlyStopping(patience=args['patience'], task_name=args['model_name'], mode=args['mode'])
        model.to(args['device'])
        stopper.load_checkpoint(model)
        model.eval()
        with torch.no_grad():
            graphs, _ = load_graphs(bin_path)
            bg = dgl.batch(graphs)
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            logits = model(bg, atom_feats)
            logits_c = logits[:, :args['classification_num']]
            logits_r = logits[:, args['classification_num']:]
            pred_c = torch.sigmoid(logits_c.detach().cpu())
            pred_r = logits_r.detach().cpu()
        pred_c_pd = pd.DataFrame(pred_c.numpy(), columns=args['classification_task'])
        pred_r_pd = pd.DataFrame(pred_r.numpy(), columns=args['regression_task'])
        pred_pd = pd.concat([pred_c_pd, pred_r_pd], axis=1)
        return pred_pd


def pred_basic_physicochemical(bin_path, gnn_name):
    assert gnn_name in ['gcn', 'gat', 'rgcn'], \
        'Expect model name to be "gcn", "gat" or "rgcn",  got {}'.format(gnn_name)
    # fix parameters of model
    args = {}
    args['device'] = "cpu"
    args['atom_data_field'] = 'atom'
    args['bond_data_field'] = 'etype'
    args['patience'] = 50
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['loop'] = True
    args['classification_task'] = []
    args['regression_task'] = ['LogS', 'LogD', 'LogP']
    args['task_number'] = len(args['classification_task']) + len(args['regression_task'])
    args['classification_num'] = len(args['classification_task'])
    if gnn_name == 'rgcn':
        args['rgcn_hidden_feats'] = [128, 128]
        args['classifier_hidden_feats'] = 128
        args['rgcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Basic_RGCN'
        basic_physicochemical_pred = pred(args, bin_path, gnn_name)
        return basic_physicochemical_pred
    if gnn_name == 'gcn':
        args['gcn_hidden_feats'] = [64, 64]
        args['classifier_hidden_feats'] = 64
        args['gcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Basic_GCN'
        basic_physicochemical_pred = pred(args, bin_path, gnn_name)
        return basic_physicochemical_pred
    if gnn_name == 'gat':
        args['gat_hidden_feats'] = [32, 32]
        args['num_heads'] = [4, 4]
        args['classifier_hidden_feats'] = 64
        args['gat_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Basic_GAT'
        basic_physicochemical_pred = pred(args, bin_path, gnn_name)
        return basic_physicochemical_pred


def pred_absorption(bin_path, gnn_name):
    assert gnn_name in ['gcn', 'gat', 'rgcn'], \
        'Expect model name to be "gcn", "gat" or "rgcn",  got {}'.format(gnn_name)
    # fix parameters of model
    args = {}
    args['device'] = "cpu"
    args['atom_data_field'] = 'atom'
    args['bond_data_field'] = 'etype'
    args['patience'] = 50
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['loop'] = True
    args['classification_task'] = ['Pgp-inh', 'Pgp-sub', 'HIA', 'F(20%)', 'F(30%)']
    args['regression_task'] = ['Caco-2', 'MDCK']
    args['task_number'] = len(args['classification_task']) + len(args['regression_task'])
    args['classification_num'] = len(args['classification_task'])
    if gnn_name == 'rgcn':
        args['rgcn_hidden_feats'] = [128, 128]
        args['classifier_hidden_feats'] = 128
        args['rgcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Absorption_RGCN'
        absorption_pred = pred(args, bin_path, gnn_name)
        return absorption_pred
    if gnn_name == 'gcn':
        args['gcn_hidden_feats'] = [64, 64]
        args['classifier_hidden_feats'] = 64
        args['gcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Absorption_GCN'
        absorption_pred = pred(args, bin_path, gnn_name)
        return absorption_pred
    if gnn_name == 'gat':
        args['gat_hidden_feats'] = [32, 32]
        args['num_heads'] = [4, 4]
        args['classifier_hidden_feats'] = 64
        args['gat_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Absorption_GAT'
        absorption_pred = pred(args, bin_path, gnn_name)
        return absorption_pred


def pred_distribution(bin_path, gnn_name):
    assert gnn_name in ['gcn', 'gat', 'rgcn'], \
        'Expect model name to be "gcn", "gat" or "rgcn",  got {}'.format(gnn_name)
    # fix parameters of model
    args = {}
    args['device'] = "cpu"
    args['atom_data_field'] = 'atom'
    args['bond_data_field'] = 'etype'
    args['patience'] = 50
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['loop'] = True
    args['classification_task'] = ['BBB']
    args['regression_task'] = ['PPB', 'VDss', 'Fu']
    args['task_number'] = len(args['classification_task']) + len(args['regression_task'])
    args['classification_num'] = len(args['classification_task'])
    if gnn_name == 'rgcn':
        args['rgcn_hidden_feats'] = [128, 128]
        args['classifier_hidden_feats'] = 128
        args['rgcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Distribution_RGCN'
        distribution_pred = pred(args, bin_path, gnn_name)
        return distribution_pred
    if gnn_name == 'gcn':
        args['gcn_hidden_feats'] = [64, 64]
        args['classifier_hidden_feats'] = 64
        args['gcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Distribution_GCN'
        distribution_pred = pred(args, bin_path, gnn_name)
        return distribution_pred
    if gnn_name == 'gat':
        args['gat_hidden_feats'] = [32, 32]
        args['num_heads'] = [4, 4]
        args['classifier_hidden_feats'] = 64
        args['gat_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Distribution_GAT'
        distribution_pred = pred(args, bin_path, gnn_name)
        return distribution_pred


def pred_metabolism(bin_path, gnn_name):
    assert gnn_name in ['gcn', 'gat', 'rgcn'], \
        'Expect model name to be "gcn", "gat" or "rgcn",  got {}'.format(gnn_name)
    # fix parameters of model
    args = {}
    args['device'] = "cpu"
    args['atom_data_field'] = 'atom'
    args['bond_data_field'] = 'etype'
    args['patience'] = 50
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['loop'] = True
    args['classification_task'] = ['CYP1A2-inh', 'CYP1A2-sub', 'CYP2C19-inh', 'CYP2C19-sub', 'CYP2C9-inh', 'CYP2C9-sub',
                                   'CYP2D6-inh',
                                   'CYP2D6-sub', 'CYP3A4-inh', 'CYP3A4-sub', 'MLM']
    args['regression_task'] = []
    args['task_number'] = len(args['classification_task']) + len(args['regression_task'])
    args['classification_num'] = len(args['classification_task'])
    if gnn_name == 'rgcn':
        args['rgcn_hidden_feats'] = [128, 128]
        args['classifier_hidden_feats'] = 128
        args['rgcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Metabolism_RGCN'
        metabolism_pred = pred(args, bin_path, gnn_name)
        return metabolism_pred
    if gnn_name == 'gcn':
        args['gcn_hidden_feats'] = [64, 64]
        args['classifier_hidden_feats'] = 64
        args['gcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Metabolism_GCN'
        metabolism_pred = pred(args, bin_path, gnn_name)
        return metabolism_pred
    if gnn_name == 'gat':
        args['gat_hidden_feats'] = [32, 32]
        args['num_heads'] = [4, 4]
        args['classifier_hidden_feats'] = 64
        args['gat_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Metabolism_GAT'
        metabolism_pred = pred(args, bin_path, gnn_name)
        return metabolism_pred


def pred_excretion_cl(bin_path, gnn_name):
    assert gnn_name in ['gcn', 'gat', 'rgcn'], \
        'Expect model name to be "gcn", "gat" or "rgcn",  got {}'.format(gnn_name)
    # fix parameters of model
    args = {}
    args['device'] = "cpu"
    args['atom_data_field'] = 'atom'
    args['bond_data_field'] = 'etype'
    args['patience'] = 50
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['loop'] = True
    args['classification_task'] = []
    args['regression_task'] = ['CL']
    args['task_number'] = len(args['classification_task']) + len(args['regression_task'])
    args['classification_num'] = len(args['classification_task'])
    if gnn_name == 'rgcn':
        args['rgcn_hidden_feats'] = [128, 128]
        args['classifier_hidden_feats'] = 128
        args['rgcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'CL_RGCN'
        cl_pred = pred(args, bin_path, gnn_name)
        return cl_pred
    if gnn_name == 'gcn':
        args['gcn_hidden_feats'] = [64, 64]
        args['classifier_hidden_feats'] = 64
        args['gcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'CL_GCN'
        cl_pred = pred(args, bin_path, gnn_name)
        return cl_pred
    if gnn_name == 'gat':
        args['gat_hidden_feats'] = [32, 32]
        args['num_heads'] = [4, 4]
        args['classifier_hidden_feats'] = 64
        args['gat_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'CL_GAT'
        cl_pred = pred(args, bin_path, gnn_name)
        return cl_pred


def pred_excretion_t12(bin_path, gnn_name):
    assert gnn_name in ['gcn', 'gat', 'rgcn'], \
        'Expect model name to be "gcn", "gat" or "rgcn",  got {}'.format(gnn_name)
    # fix parameters of model
    args = {}
    args['device'] = "cpu"
    args['atom_data_field'] = 'atom'
    args['bond_data_field'] = 'etype'
    args['patience'] = 50
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['loop'] = True
    args['classification_task'] = ['T12']
    args['regression_task'] = []
    args['task_number'] = len(args['classification_task']) + len(args['regression_task'])
    args['classification_num'] = len(args['classification_task'])
    if gnn_name == 'rgcn':
        args['rgcn_hidden_feats'] = [128, 128]
        args['classifier_hidden_feats'] = 128
        args['rgcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'T12_RGCN'
        t12_pred = pred(args, bin_path, gnn_name)
        return t12_pred
    if gnn_name == 'gcn':
        args['gcn_hidden_feats'] = [64, 64]
        args['classifier_hidden_feats'] = 64
        args['gcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'T12_GCN'
        t12_pred = pred(args, bin_path, gnn_name)
        return t12_pred
    if gnn_name == 'gat':
        args['gat_hidden_feats'] = [32, 32]
        args['num_heads'] = [4, 4]
        args['classifier_hidden_feats'] = 64
        args['gat_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'T12_GAT'
        t12_pred = pred(args, bin_path, gnn_name)
        return t12_pred


def pred_toxicity(bin_path, gnn_name):
    assert gnn_name in ['gcn', 'gat', 'rgcn'], \
        'Expect model name to be "gcn", "gat" or "rgcn",  got {}'.format(gnn_name)
    # fix parameters of model
    args = {}
    args['device'] = "cpu"
    args['atom_data_field'] = 'atom'
    args['bond_data_field'] = 'etype'
    args['patience'] = 50
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['loop'] = True
    args['classification_task'] = ['hERG', 'H-HT', 'DILI', 'Ames', 'ROA', 'FDAMDD', 'SkinSen', 'Carcinogenicity', 'EC',
                                   'EI',
                                   'Respiratory']
    args['regression_task'] = ['BCF', 'IGC50', 'LC50', 'LC50DM']
    args['task_number'] = len(args['classification_task']) + len(args['regression_task'])
    args['classification_num'] = len(args['classification_task'])
    if gnn_name == 'rgcn':
        args['rgcn_hidden_feats'] = [128, 128]
        args['classifier_hidden_feats'] = 128
        args['rgcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Toxicity_RGCN'
        toxicity_pred = pred(args, bin_path, gnn_name)
        return toxicity_pred
    if gnn_name == 'gcn':
        args['gcn_hidden_feats'] = [64, 64]
        args['classifier_hidden_feats'] = 64
        args['gcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Toxicity_GCN'
        toxicity_pred = pred(args, bin_path, gnn_name)
        return toxicity_pred
    if gnn_name == 'gat':
        args['gat_hidden_feats'] = [32, 32]
        args['num_heads'] = [4, 4]
        args['classifier_hidden_feats'] = 64
        args['gat_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Toxicity_GAT'
        toxicity_pred = pred(args, bin_path, gnn_name)
        return toxicity_pred


def pred_tox21(bin_path, gnn_name):
    assert gnn_name in ['gcn', 'gat', 'rgcn'], \
        'Expect model name to be "gcn", "gat" or "rgcn",  got {}'.format(gnn_name)
    # fix parameters of model
    args = {}
    args['device'] = "cpu"
    args['atom_data_field'] = 'atom'
    args['bond_data_field'] = 'etype'
    args['patience'] = 50
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['loop'] = True
    args['classification_task'] = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                                   'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                                   'SR-HSE', 'SR-MMP', 'SR-p53']
    args['regression_task'] = []
    args['task_number'] = len(args['classification_task']) + len(args['regression_task'])
    args['classification_num'] = len(args['classification_task'])
    if gnn_name == 'rgcn':
        args['rgcn_hidden_feats'] = [128, 128]
        args['classifier_hidden_feats'] = 128
        args['rgcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Tox21_RGCN'
        tox21_pred = pred(args, bin_path, gnn_name)
        return tox21_pred
    if gnn_name == 'gcn':
        args['gcn_hidden_feats'] = [64, 64]
        args['classifier_hidden_feats'] = 64
        args['gcn_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Tox21_GCN'
        tox21_pred = pred(args, bin_path, gnn_name)
        return tox21_pred
    if gnn_name == 'gat':
        args['gat_hidden_feats'] = [32, 32]
        args['num_heads'] = [4, 4]
        args['classifier_hidden_feats'] = 64
        args['gat_drop_out'] = 0.2
        args['drop_out'] = 0.2
        args['model_name'] = 'Tox21_GAT'
        tox21_pred = pred(args, bin_path, gnn_name)
        return tox21_pred


def pred_admet(smiles_list, cache_bin_path, cache_csv_path, result_path, gnn_name):
    structure_list = smiles_list['structure_id'].tolist()
    transformations = smiles_list['transformation'].tolist()
    left_fragment = smiles_list['left_fragment'].tolist()
    right_fragment = smiles_list['right_fragment'].tolist()
    flag = smiles_list['flag'].tolist()
    smiles_list = smiles_list['smiles'].tolist()
    
    smiles2graphs_save(smiles_list, bin_path=cache_bin_path, csv_path=cache_csv_path)
    basic_physicochemical = pred_basic_physicochemical(cache_bin_path, gnn_name)
    absorption = pred_absorption(cache_bin_path, gnn_name)
    # absorption_mdck = pred_absorption_mdck(cache_bin_path, gnn_name)
    distribution = pred_distribution(cache_bin_path, gnn_name)
    # distribution_vdss = pred_distribution_vdss(cache_bin_path, gnn_name)
    metabolism = pred_metabolism(cache_bin_path, gnn_name)
    excretion_cl = pred_excretion_cl(cache_bin_path, gnn_name)
    excretion_t12 = pred_excretion_t12(cache_bin_path, gnn_name)
    toxcity = pred_toxicity(cache_bin_path, gnn_name)
    tox21 = pred_tox21(cache_bin_path, gnn_name)
    medchem = MedicinalChemistry([Chem.MolFromSmiles(item) for item in smiles_list])
    sascore = medchem.CalculateSAscore()

    # absorption["MDCK"] = 10 ** (absorption.MDCK.values)
    distribution["Fu"] = 10 ** -(distribution.Fu.values)

    admet_pred = pd.concat(
        [pd.DataFrame(columns=['structure_id', 'smiles']), basic_physicochemical, absorption, distribution, metabolism,
         excretion_cl,
         excretion_t12, toxcity, tox21], axis=1)
    admet_pred['HIA'] = 1 - admet_pred['HIA']
    admet_pred['F(20%)'] = 1 - admet_pred['F(20%)']
    admet_pred['F(30%)'] = 1 - admet_pred['F(30%)']
    admet_pred['Fu'] = admet_pred['Fu'] * 100
    admet_pred["MDCK"] = 10 ** (absorption.MDCK.values)
    new_smiles_pd = pd.read_csv(cache_csv_path, index_col=None)
    new_smiles_list = new_smiles_pd['smiles'].values.tolist()
    admet_pred['structure_id'] = structure_list
    admet_pred['smiles'] = new_smiles_list
    admet_pred['transformation'] = transformations
    admet_pred['left_fragment'] = left_fragment
    admet_pred['right_fragment'] = right_fragment
    admet_pred['flag'] = flag
    admet_pred['sascore'] = sascore
    admet_pred = admet_pred.round(7)
    # 删除MLM列
    admet_pred = admet_pred.drop(columns=['MLM'])

    admet_pred.to_csv(result_path, index_label='mol_index')
    # return len(new_smiles_list), atom, medRes
    return len(new_smiles_list), admet_pred


# 毒性新增
class Toxicity:

    def __init__(self, mols):

        self.mols = mols if isinstance(mols, Iterable) else (mols,)
        self.smis = None

    def ScreenToxicityFragments(self):
        # print(self.mols)
        res = pd.DataFrame()
        atoms = pd.DataFrame()
        files = os.listdir(SmartDir)

        for file in files:
            endpoint = os.path.splitext(file)[0]

            length = []
            a = []
            if endpoint not in ["Alarm_NMR", "BMS", "Chelating", "PAINS"]:
                for x in SubstrutureFilter().screening(self.mols, endpoint):
                    leng = 0 if x["MatchedNames"] == ['-'] else len(x["MatchedNames"])
                    length.append(leng)
                    a.append([x["MatchedAtoms"]])
                res[f"{endpoint}"] = length
                # print(a)
                atoms[f"{endpoint}"] = a
        return res, atoms

    def CalculateAllToxicityProperties(self):
        # T = self.CalculateToxicityProperties()
        # T21 = self.CalculateToxicity21Properties()
        Tsub = self.ScreenToxicityFragments()

        # return T, T21, Tsub
        return Tsub


class Basic:

    def __init__(self, mols):
        self.mols = mols if isinstance(mols, Iterable) else (mols,)
        self.smis = None

    def CalculateMolWeight(self):
        """
        Calculation of molecular weight(contain hydrogen atoms)
        --->MW

        :param mol: molecule
        :type mol: rdkit.Chem.rdchem.Mol
        :return: the weight of molecule(contain hydrogen atoms)
        :rtype: float

        """
        MW = np.array(list(map(Descriptors.ExactMolWt, self.mols)))
        return np.round(MW, 2)

    def CalculateSol(self):
        """
        Calculation of molecular logS, logD and logP
        --->logS, logD and logP

        :param mol: molecular
        :type mol: rdkit.Chem.rdchem.Mol
        :return: molecular logP
        :rtype: float

        """
        if self.smis is None:
            self.smis = [Chem.MolToSmiles(mol) for mol in self.mols]

        res = pred_admet(self.smis)
        return res

    def CalculateNumHAcceptors(self):
        """
        Caculation of the number of Hydrogen Bond Acceptors
        --->nHA

        :param mol: molecular
        :type mol: rdkit.Chem.rdchem.Mol
        :return: the number of Hydrogen Bond Acceptors
        :rtype: int

        """
        nHA = np.array(list(map(Lipinski.NOCount, self.mols)))
        return nHA

    def CalculateNumHDonors(self):
        """
        Caculation of the Number of Hydrogen Bond Donors
        --->nHD

        :param mol: molecular
        :type mol: rdkit.Chem.rdchem.Mol
        :return: the number of Hydrogen Bond Donors
        :rtype: int

        """
        # nHD = np.array(list(map(Lipinski.NumHDonors, self.mols)))
        nHD = np.array(list(map(Lipinski.NHOHCount, self.mols)))
        return nHD

    def CalculateTPSA(self):
        """
        Calculation of TPSA
        --->TPSA

        :param mol: molecular
        :type mol: rdkit.Chem.rdchem.Mol
        :return: TPSA
        :rtype: float

        """
        TPSA = np.array(list(map(Descriptors.TPSA, self.mols)))
        return np.round(TPSA, 2)

    def CalculateNumRotatableBonds(self):
        """
        Calculation of the number of rotatable Bonds
        --->nRot

        Note:
            In some situaion Amide C-N bonds are not considered
            because of their high rotational energy barrier

        :param mol: molecule
        :type mol: rdkit.Chem.rdchem.Mol
        :return: the number of rotatableBond
        :rtype: int


        """
        nRot = []
        patt = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
        for mol in self.mols:
            nRot.append(len(mol.GetSubstructMatches(patt)))
        return np.array(nRot)

    def CalculateNumBonds(self):
        """
        Calculation the number of bonds where between heavy atoms
        --->nBond

        :param mol: molecule
        :type mol: rdkit.Chem.rdchem.Mol
        :return: the number of bonds where between heavy atoms
        :rtype: int

        """
        nBond = [mol.GetNumBonds() for mol in self.mols]
        return np.array(nBond)

    def CalculateNumRigidBonds(self):
        """
        Number of non-flexible bonds, in opposite to rotatable bonds
        --->nRig

        :param mol: molecule
        :type mol: rdkit.Chem.rdchem.Mol
        :return: the number of non-flexible bonds
        :rtype: int

        """
        nBOND = self.CalculateNumBonds()
        nFlex = []
        for mol in self.mols:
            flex = 0
            for bond in mol.GetBonds():
                bondtype = bond.GetBondType()
                if bondtype == Chem.rdchem.BondType.SINGLE and not bond.IsInRing():
                    flex += 1
            nFlex.append(flex)

        nFlex = np.array(nFlex)
        # print(nBOND)
        # print(nFlex)
        nRig = nBOND - nFlex
        # print(nRig)
        return nRig

    def CalculateNumRing(self):
        """
        Calculation of the number of ring
        --->nRing

        :param mol: molecule
        :type mol: rdkit.Chem.rdchem.Mol
        :return: the number of ring
        :rtype: int

        """
        nRing = [Chem.GetSSSR(mol) for mol in self.mols]
        return np.array(nRing)

    def CalculateMaxSizeSystemRing(self):
        """
        Number of atoms involved in the biggest system ring
        ---> maxring

        :param mol: molecular
        :type mol: rdkit.Chem.rdchem.Mol
        :return: number of atoms involved in the biggest system ring
        :rtype: int

        """
        MaxRing = []

        for mol in self.mols:
            # print(mol)
            # 0.Get the scaffold
            core = MurckoScaffold.GetScaffoldForMol(mol)
            fw = MurckoScaffold.MakeScaffoldGeneric(core)
            # 1.Obtaining which atoms consist of rings
            Max = 0
            ri = fw.GetRingInfo()
            atoms = list(ri.AtomRings())
            length = len(atoms)
            if length == 0:
                MaxRing.append(0)
            else:
                rw = Chem.RWMol(fw)
                # 2.Judge which atoms are replacement
                atoms = [set(x) for x in atoms]
                for pair in combinations(range(length), 2):
                    replace = list(atoms[pair[0]] & atoms[pair[1]])
                    if len(replace) >= 2:
                        for repl in list(combinations(replace, 2)):
                            rw.RemoveBond(*repl)
                    else:
                        pass
                m = Chem.MolFromSmiles(Chem.MolToSmiles(rw))
                ri = m.GetRingInfo()
                bonds = ri.BondRings()
                for item in bonds:
                    if len(item) > Max:
                        Max = len(item)
                MaxRing.append(Max)

        return np.array(MaxRing)

    def CalculateNumCarbon(self):
        """
        Calculation of Carbon number in a molecule
        --->nC

        :param mol: molecular
        :type mol: rdkit.Chem.rdchem.Mol
        :return: the number of carbon atoms
        :rtype: int

        """
        nC = []
        for mol in self.mols:
            nC.append(len(
                [atom for atom in mol.GetAtoms()
                 if atom.GetAtomicNum() == 6]
            ))
        return np.array(nC)

    def CalculateNumHetero(self):
        """
        Calculation of the number of heteroatom in a molecule
        --->nHet

        :param mol: molecule
        :type mol: rdkit.Chem.rdchem.Mol
        :return: the number of heteroatom in a molecule
        :rtype: int

        """
        nHet = []
        for mol in self.mols:
            i = len(
                [atom for atom in mol.GetAtoms()
                 if atom.GetAtomicNum() in [1, 6]]
            )
            nHet.append(mol.GetNumAtoms() - i)
        return np.array(nHet)

    def CalculateHetCarbonRatio(self):
        """
        The ratio between the number of non carbon atoms and the number of carbon atoms.
        --->HetRatio

        :param mol: molecular
        :type mol: rdkit.Chem.rdchem.Mol
        :return: the ratio between the number of non carbon atoms and the number of carbon atoms
        :rtype: float

        """
        nHet = self.CalculateNumHetero()
        nCarb = self.CalculateNumCarbon()
        return np.round(nHet / nCarb, 2)

    def CalculateMolFCharge(self):
        """
        Calculation of formal charge of molecule
        --->fChar

        :param mol: molecular
        :type mol: rdkit.Chem.rdchem.Mol
        :return: formal charge of molecule
        :rtype: float

        """
        FChar = []
        for mol in self.mols:
            mol = Chem.AddHs(mol)
            FChar.append(sum([atom.GetFormalCharge()
                              for atom in mol.GetAtoms()]))
        return np.array(FChar)

    def CalculateMolVolume(self):
        """
        Calculation of Van der Waals Volume of molecule
        --->MV

        Equation:
            for single atom: Vw = 4/3*pi*rw^3, the rw is the Van der Waals radius of atom
            VvdW = ∑(atom contributions)-5.92NB(Unit in Å^3), NB is the total number of bonds
            the Van der Waals radius of atom is derived from wikipedia.

        :param mol: molecular
        :type mol: rdkit.Chem.rdchem.Mol
        :return: Van der Waals Volume of molecule
        :rtype: float

        """
        vols = []
        Radii = {'H': 1.20, 'C': 1.70, 'N': 1.55,
                 'O': 1.52, 'S': 1.80, 'P': 1.80,
                 'F': 1.47, 'Cl': 1.75, 'Br': 1.85,
                 'I': 1.98, 'Na': 2.27, 'Mg': 1.73,
                 'K': 2.75, 'Ca': 2.31, 'Ba': 2.68,
                 'He': 140, 'Li': 182, 'Be': 153,
                 'B': 192, 'Ne': 154, 'Al': 184,
                 'Si': 210, 'Ar': 188, 'Ni': 163,
                 'Cu': 140, 'Zn': 139, 'Ga': 187,
                 'Ge': 211, 'As': 185, 'Se': 190,
                 'Kr': 202, 'Rb': 303, 'Sr': 249,
                 'Pd': 163, 'Ag': 172, 'Cd': 158,
                 'In': 193, 'Sn': 217, 'Sb': 206,
                 'Te': 206, 'Xe': 216, 'Cs': 343,
                 'Pt': 175, 'Au': 166, 'U': 186,
                 'Hg': 155, 'Tl': 196, 'Pb': 202,
                 'Bi': 207, 'Po': 197, 'At': 202,
                 'Rn': 220, 'Fr': 348, 'Ra': 283}
        for mol in self.mols:
            mol = Chem.AddHs(mol)
            contrib = []
            for atom in mol.GetAtoms():
                try:
                    contrib.append(Radii[atom.GetSymbol()])
                except:
                    pass
            # contrib = [Radii[atom.GetSymbol()] for atom in mol.GetAtoms()]
            contrib = [pi * (r ** 3) * 4 / 3 for r in contrib]
            vol = sum(contrib) - 5.92 * len(mol.GetBonds())
            vols.append(vol)
        return np.array(vols)

    def CalculatepKa(self, logDs, logPs):
        """
        Calculating pKa based on the ralation between logD and logP in specific pH.
        --->pKa

        Eq.:
            abs(pH-pKa) = log10(10^(logP-logD)-1)
            pKa = pH - log10(10^(logP-logD)-1) for acid
            pKa = log10(10^(logP-logD)-1) - pH for base

        :param mol: molecular
        :type mol: rdkit.Chem.rdchem.Mol
        :return: molecular pKa
        :rtype: float

        """
        pKas = []
        acid_fragment = [
            '[!H0;F,Cl,Br,I,N+,$([OH]-*=[!#6]),+]',
            '[CX3](=O)[OX2H1]',
            '[CX3](=O)[OX1H0-,OX2H1]',
            '[$([OH]-*=[!#6])]',
            '[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]',
            '[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]',
            '[CX3](=[OX1])[F,Cl,Br,I]'
        ]
        for mol, logD, logP in zip(self.mols, logDs, logPs):
            for sma in acid_fragment:
                patt = Chem.MolFromSmarts(sma)
                if mol.HasSubstructMatch(patt):
                    status = 'acid'
                    break
            else:
                status = 'base'

            try:
                if status == 'acid':
                    pKa = 7.4 - log10(10 ** (logP - logD) - 1)
                else:
                    pKa = log10(10 ** (logP - logD) - 1) - 7.4
                pKas.append(pKa)
            except:
                pKas.append(np.float('nan'))
        return np.array(pKas)

    def CalculateNumStereocenters(self):
        """
        the number of stereo centers
        --->nStereo

        :param mol: molecular
        :type mol: rdkit.Chem.rdchem.Mol
        :return: the number of stereo centers
        :rtype: int

        """
        nStereos = []
        for mol in self.mols:
            nStereos.append(Chem.CalcNumAtomStereoCenters(mol))
        return nStereos

    def CalculateBasicProperties(self):

        MW = self.CalculateMolWeight()
        Vol = self.CalculateMolVolume()
        Dense = MW / Vol
        nHA = self.CalculateNumHAcceptors()
        nHD = self.CalculateNumHDonors()
        TPSA = self.CalculateTPSA()
        nRot = self.CalculateNumRotatableBonds()
        nRing = self.CalculateNumRing()
        MaxRing = self.CalculateMaxSizeSystemRing()
        # nC = self.CalculateNumCarbon()
        nHet = self.CalculateNumHetero()
        # HetRtio = self.CalculateHetCarbonRatio()
        fChar = self.CalculateMolFCharge()
        nRig = self.CalculateNumRigidBonds()
        Flex = nRot / nRig
        nStereo = self.CalculateNumStereocenters()

        # sol = pd.concat(self.CalculateSol(), axis=1)

        out = pd.DataFrame(
            {
                "MW": MW,
                "Vol": Vol,
                "Dense": Dense,
                "nHA": nHA,
                "nHD": nHD,
                "TPSA": TPSA,
                "nRot": nRot,
                "nRing": nRing,
                "MaxRing": MaxRing,
                # "nC": nC,
                "nHet": nHet,
                # "HetRatio": HetRtio,
                "fChar": fChar,
                "nRig": nRig,
                "Flex": Flex,
                "nStereo": nStereo
            },
        )

        # out = pd.concat([out, sol], axis=1)
        out = pd.concat([out, ], axis=1)
        # out["pKa"] = self.CalculatepKa(out.LogD, out.LogP)
        return out

