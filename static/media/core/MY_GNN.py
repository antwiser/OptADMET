import datetime
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score
import torch
import torch.nn.functional as F
import numpy as np
import random
from dgl.readout import sum_nodes
from dgl.nn.pytorch.conv import RelGraphConv, GraphConv, GATConv
from torch import nn

import os
from django.conf import settings


class WeightAndSum(nn.Module):
    """Compute importance weights for atoms and perform a weighted sum.

    Parameters
    ----------
    in_feats : int
        Input atom feature size

    return_weight: bool
        Defalt: False
    """
    def __init__(self, in_feats, task_num=1, attention=True, return_weight=False):
        super(WeightAndSum, self).__init__()
        self.attention = attention
        self.in_feats = in_feats
        self.task_num = task_num
        self.return_weight=return_weight
        self.atom_weighting_specific = nn.ModuleList([self.atom_weight(self.in_feats) for _ in range(self.task_num)])
        self.shared_weighting = self.atom_weight(self.in_feats)
    def forward(self, bg, feats):
        """Compute molecule representations out of atom representations
        Parameters
        ----------
        bg : BatchedDGLGraph
            B Batched DGLGraphs for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, self.in_feats)
            Representations for all atoms in the molecules
            * N is the total number of atoms in all molecules
        Returns
        -------
        FloatTensor of shape (B, self.in_feats)
            Representations for B molecules
        atom_weight for each atom
        """
        feat_list = []
        atom_list = []
        # cal specific feats
        for i in range(self.task_num):
            with bg.local_scope():
                bg.ndata['h'] = feats
                weight = self.atom_weighting_specific[i](feats)
                bg.ndata['w'] = weight
                specific_feats_sum = sum_nodes(bg, 'h', 'w')
                atom_list.append(bg.ndata['w'])
            feat_list.append(specific_feats_sum)

        # cal shared feats
        with bg.local_scope():
            bg.ndata['h'] = feats
            bg.ndata['w'] = self.shared_weighting(feats)
            shared_feats_sum = sum_nodes(bg, 'h', 'w')
        # feat_list.append(shared_feats_sum)
        if self.attention:
            if self.return_weight:
                return feat_list, atom_list
            else:
                return feat_list
        else:
            return shared_feats_sum

    def atom_weight(self, in_feats):
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
            )


class GCNLayer(nn.Module):
    """Single layer GCN for updating node features

    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features
    activation : activation function
        Default to be ReLU
    residual : bool
        Whether to use residual connection, default to be True
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """
    def __init__(self, in_feats, out_feats, activation=F.relu,
                 residual=True, batchnorm=True, dropout=0.):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats,
                                    norm='none', activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, g, feats):
        """Update atom representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph with batch size B for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization

        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        new_feats = self.graph_conv(g, feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class GATLayer(nn.Module):
    """Single layer GAT for updating node features

    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features for each attention head
    num_heads : int
        Number of attention heads
    feat_drop : float
        Dropout applied to the input features
    attn_drop : float
        Dropout applied to attention values of edges
    alpha : float
        Hyperparameter in LeakyReLU, slope for negative values. Default to be 0.2
    residual : bool
        Whether to perform skip connection, default to be False
    agg_mode : str
        The way to aggregate multi-head attention results, can be either
        'flatten' for concatenating all head results or 'mean' for averaging
        all head results
    activation : activation function or None
        Activation function applied to aggregated multi-head results, default to be None.
    """
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=True, agg_mode='flatten', activation=None):
        super(GATLayer, self).__init__()
        self.gnn = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads,
                           feat_drop=feat_drop, attn_drop=attn_drop,
                           negative_slope=alpha, residual=residual)
        assert agg_mode in ['flatten', 'mean']
        self.agg_mode = agg_mode
        self.activation = activation

    def forward(self, bg, feats):
        """Update atom representations

        Parameters
        ----------
        bg : DGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization

        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size. If self.agg_mode == 'flatten', this would
              be out_feats * num_heads, else it would be just out_feats.
        """
        new_feats = self.gnn(bg, feats)
        if self.agg_mode == 'flatten':
            new_feats = new_feats.flatten(1)
        else:
            new_feats = new_feats.mean(1)

        if self.activation is not None:
            new_feats = self.activation(new_feats)

        return new_feats


class RGCNLayer(nn.Module):
    """Single layer H-RGCN for updating node features
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features
    num_rels: int
        Number of bond type
    activation : activation function
        Default to be ReLU
    loop: bool:
        Whether to use self loop
        Default to be False
    residual : bool
        Whether to use residual connection, default to be True
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True
    rgcn_drop_out : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    """
    def __init__(self, in_feats, out_feats, num_rels=64*21, activation=F.relu, loop=False,
                 residual=True, batchnorm=True, rgcn_drop_out=0.5):
        super(RGCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv_layer = RelGraphConv(in_feats, out_feats, num_rels=num_rels, regularizer='basis',
                                               num_bases=None, bias=True, activation=activation,
                                               self_loop=loop, dropout=rgcn_drop_out)
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, node_feats, etype, norm=None):
        """Update atom representations
        Parameters
        ----------
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        node_feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization
        etype: int
            bond type
        norm: torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`
        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        new_feats = self.graph_conv_layer(bg, node_feats, etype, norm)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        del res_feats
        torch.cuda.empty_cache()
        return new_feats


class BaseGNN(nn.Module):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    gnn_out_feats : int
        Number of atom representation features after using GNN
    len_descriptors : int
        length of descriptors
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    rgcn_drop_out: float
        dropout rate for HRGCN layer
    n_tasks : int
        Number of prediction tasks
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    return_weight: bool
        Wether to return atom weight defalt=False
    """
    def __init__(self, gnn_out_feats, n_tasks, rgcn_drop_out=0.5, return_mol_embedding=False, return_weight=False,
                 classifier_hidden_feats=128, dropout=0.):
        super(BaseGNN, self).__init__()
        self.task_num = n_tasks
        self.gnn_layers = nn.ModuleList()
        self.return_weight = return_weight
        self.weighted_sum_readout = WeightAndSum(gnn_out_feats, self.task_num, return_weight=self.return_weight)
        self.fc_in_feats = gnn_out_feats
        self.return_mol_embedding=return_mol_embedding

        self.fc_layers1 = nn.ModuleList([self.fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers3 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])

        self.output_layer1 = nn.ModuleList(
            [self.output_layer(classifier_hidden_feats, 1) for _ in range(self.task_num)])

    def forward(self, bg, node_feats, etype, norm=None):
        """Multi-task prediction for a batch of molecules
        Parameters
        ----------
        bg : BatchedDGLGraph
            B Batched DGLGraphs for processing multiple molecules in parallel
        node_feats : FloatTensor of shape (N, M0)
            Initial features for all atoms in the batch of molecules
        etype: int
            bond type
        descriptors: FloatTensor for shape (M1, M2)
            descriptor used as global feature, M1= number of molecule, M2 = Length of descriptors
        norm: torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            Soft prediction for all tasks on the batch of molecules·
        """
        # Update atom features with GNNs
        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats, etype, norm)

        # Compute molecule features from atom features
        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)

        for i in range(self.task_num):
            # mol_feats = torch.cat([feats_list[-1], feats_list[i]], dim=1)
            mol_feats = feats_list[i]
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)
            predict = self.output_layer1[i](h3)
            if i == 0:
                prediction_all = predict
            else:
                prediction_all = torch.cat([prediction_all, predict], dim=1)
        # generate toxicity fingerprints
        if self.return_mol_embedding:
            return feats_list[0]
        else:
            # generate atom weight
            if self.return_weight:
                return prediction_all, atom_weight_list
            # just generate prediction
            return prediction_all

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats)
                )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
                )


class MTRGCN(BaseGNN):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    Rgcn_hidden_feats : list of int
        rgcn_hidden_feats[i] gives the number of output atom features
        in the i+1-th HRGCN layer
    n_tasks : int
        Number of prediction tasks
    len_descriptors : int
        length of descriptors
    return_weight : bool
        Wether to return weight
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    is_descriptor: bool
        Wether to use descriptor
    loop : bool
        Wether to use self loop
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    rgcn_drop_rate : float
        The probability for dropout of HRGCN layer. Default to be 0.5
    dropout : float
        The probability for dropout of MLP layer. Default to be 0.
    """
    def __init__(self, in_feats, rgcn_hidden_feats, n_tasks, return_weight=False,
                 classifier_hidden_feats=128, loop=False, return_mol_embedding=False,
                 rgcn_drop_out=0.5, dropout=0.):
        super(MTRGCN, self).__init__(gnn_out_feats=rgcn_hidden_feats[-1],
                                     n_tasks=n_tasks,
                                     classifier_hidden_feats=classifier_hidden_feats,
                                     return_mol_embedding=return_mol_embedding,
                                     return_weight=return_weight,
                                     rgcn_drop_out=rgcn_drop_out,
                                     dropout=dropout,
                                     )

        for i in range(len(rgcn_hidden_feats)):
            out_feats = rgcn_hidden_feats[i]
            self.gnn_layers.append(RGCNLayer(in_feats, out_feats, loop=loop, rgcn_drop_out=rgcn_drop_out))
            in_feats = out_feats


class BaseGCN(nn.Module):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    gnn_out_feats : int
        Number of atom representation features after using GNN
    len_descriptors : int
        length of descriptors
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    rgcn_drop_out: float
        dropout rate for HRGCN layer
    n_tasks : int
        Number of prediction tasks
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    return_weight: bool
        Wether to return atom weight defalt=False
    """
    def __init__(self, gnn_out_feats, n_tasks, gcn_drop_out=0.5, return_mol_embedding=False, return_weight=False,
                 classifier_hidden_feats=128, dropout=0.):
        super(BaseGCN, self).__init__()
        self.task_num = n_tasks
        self.gnn_layers = nn.ModuleList()
        self.return_weight = return_weight
        self.weighted_sum_readout = WeightAndSum(gnn_out_feats, self.task_num, return_weight=self.return_weight)
        self.fc_in_feats = gnn_out_feats
        self.return_mol_embedding=return_mol_embedding

        self.fc_layers1 = nn.ModuleList([self.fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers3 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])

        self.output_layer1 = nn.ModuleList(
            [self.output_layer(classifier_hidden_feats, 1) for _ in range(self.task_num)])

    def forward(self, bg, node_feats):
        """Multi-task prediction for a batch of molecules
        Parameters
        ----------
        bg : BatchedDGLGraph
            B Batched DGLGraphs for processing multiple molecules in parallel
        node_feats : FloatTensor of shape (N, M0)
            Initial features for all atoms in the batch of molecules
        etype: int
            bond type
        descriptors: FloatTensor for shape (M1, M2)
            descriptor used as global feature, M1= number of molecule, M2 = Length of descriptors
        norm: torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            Soft prediction for all tasks on the batch of molecules·
        """
        # Update atom features with GNNs
        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats)

        # Compute molecule features from atom features
        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)

        for i in range(self.task_num):
            # mol_feats = torch.cat([feats_list[-1], feats_list[i]], dim=1)
            mol_feats = feats_list[i]
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)
            predict = self.output_layer1[i](h3)
            if i == 0:
                prediction_all = predict
            else:
                prediction_all = torch.cat([prediction_all, predict], dim=1)
        # generate toxicity fingerprints
        if self.return_mol_embedding:
            return feats_list[0]
        else:
            # generate atom weight
            if self.return_weight:
                return prediction_all, atom_weight_list
            # just generate prediction
            return prediction_all

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats)
                )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
                )


class MTGCN(BaseGCN):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    Rgcn_hidden_feats : list of int
        rgcn_hidden_feats[i] gives the number of output atom features
        in the i+1-th HRGCN layer
    n_tasks : int
        Number of prediction tasks
    len_descriptors : int
        length of descriptors
    return_weight : bool
        Wether to return weight
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    is_descriptor: bool
        Wether to use descriptor
    loop : bool
        Wether to use self loop
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    rgcn_drop_rate : float
        The probability for dropout of HRGCN layer. Default to be 0.5
    dropout : float
        The probability for dropout of MLP layer. Default to be 0.
    """
    def __init__(self, in_feats, gcn_hidden_feats, n_tasks, return_weight=False,
                 classifier_hidden_feats=128, return_mol_embedding=False,
                 gcn_drop_out=0.5, dropout=0.):
        super(MTGCN, self).__init__(gnn_out_feats=gcn_hidden_feats[-1],
                                    n_tasks=n_tasks,
                                    classifier_hidden_feats=classifier_hidden_feats,
                                    return_mol_embedding=return_mol_embedding,
                                    return_weight=return_weight,
                                    gcn_drop_out=gcn_drop_out,
                                    dropout=dropout,
                                    )

        for i in range(len(gcn_hidden_feats)):
            out_feats = gcn_hidden_feats[i]
            self.gnn_layers.append(GCNLayer(in_feats, out_feats, dropout=gcn_drop_out))
            in_feats = out_feats


class BaseGAT(nn.Module):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    gnn_out_feats : int
        Number of atom representation features after using GNN
    len_descriptors : int
        length of descriptors
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    rgcn_drop_out: float
        dropout rate for HRGCN layer
    n_tasks : int
        Number of prediction tasks
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    return_weight: bool
        Wether to return atom weight defalt=False
    """
    def __init__(self, gat_out_feats, n_tasks, gat_drop_out=0.5, return_mol_embedding=False, return_weight=False,
                 classifier_hidden_feats=128, dropout=0.):
        super(BaseGAT, self).__init__()
        self.task_num = n_tasks
        self.gnn_layers = nn.ModuleList()
        self.return_weight = return_weight
        self.weighted_sum_readout = WeightAndSum(gat_out_feats, self.task_num, return_weight=self.return_weight)
        self.fc_in_feats = gat_out_feats
        self.return_mol_embedding=return_mol_embedding

        self.fc_layers1 = nn.ModuleList([self.fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers3 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])

        self.output_layer1 = nn.ModuleList(
            [self.output_layer(classifier_hidden_feats, 1) for _ in range(self.task_num)])

    def forward(self, bg, node_feats):
        """Multi-task prediction for a batch of molecules
        Parameters
        ----------
        bg : BatchedDGLGraph
            B Batched DGLGraphs for processing multiple molecules in parallel
        node_feats : FloatTensor of shape (N, M0)
            Initial features for all atoms in the batch of molecules
        etype: int
            bond type
        descriptors: FloatTensor for shape (M1, M2)
            descriptor used as global feature, M1= number of molecule, M2 = Length of descriptors
        norm: torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            Soft prediction for all tasks on the batch of molecules·
        """
        # Update atom features with GNNs
        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats)

        # Compute molecule features from atom features
        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)

        for i in range(self.task_num):
            # mol_feats = torch.cat([feats_list[-1], feats_list[i]], dim=1)
            mol_feats = feats_list[i]
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)
            predict = self.output_layer1[i](h3)
            if i == 0:
                prediction_all = predict
            else:
                prediction_all = torch.cat([prediction_all, predict], dim=1)
        # generate toxicity fingerprints
        if self.return_mol_embedding:
            return feats_list[0]
        else:
            # generate atom weight
            if self.return_weight:
                return prediction_all, atom_weight_list
            # just generate prediction
            return prediction_all

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats)
                )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
                )


class MTGAT(BaseGAT):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    Rgcn_hidden_feats : list of int
        rgcn_hidden_feats[i] gives the number of output atom features
        in the i+1-th HRGCN layer
    n_tasks : int
        Number of prediction tasks
    len_descriptors : int
        length of descriptors
    return_weight : bool
        Wether to return weight
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    is_descriptor: bool
        Wether to use descriptor
    loop : bool
        Wether to use self loop
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    rgcn_drop_rate : float
        The probability for dropout of HRGCN layer. Default to be 0.5
    dropout : float
        The probability for dropout of MLP layer. Default to be 0.
    """
    def __init__(self, in_feats, gat_hidden_feats, n_tasks, num_heads, return_weight=False,
                 classifier_hidden_feats=128, return_mol_embedding=False,
                 gat_drop_out=0.5, dropout=0.):
        super(MTGAT, self).__init__(gat_out_feats=gat_hidden_feats[-1],
                                    n_tasks=n_tasks,
                                    classifier_hidden_feats=classifier_hidden_feats,
                                    return_mol_embedding=return_mol_embedding,
                                    return_weight=return_weight,
                                    gat_drop_out=gat_drop_out,
                                    dropout=dropout,
                                    )
        assert len(gat_hidden_feats) == len(num_heads), \
            'Got gat_hidden_feats with length {:d} and num_heads with length {:d}, ' \
            'expect them to be the same.'.format(len(gat_hidden_feats), len(num_heads))
        num_layers = len(num_heads)
        for l in range(num_layers):
            if l > 0:
                in_feats = gat_hidden_feats[l - 1] * num_heads[l - 1]

            if l == num_layers - 1:
                agg_mode = 'mean'
                agg_act = None
            else:
                agg_mode = 'flatten'
                agg_act = F.elu

            self.gnn_layers.append(GATLayer(in_feats, gat_hidden_feats[l], num_heads[l],
                                            feat_drop=gat_drop_out, attn_drop=gat_drop_out,
                                            agg_mode=agg_mode, activation=agg_act))


def set_random_seed(seed=10):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class EarlyStopping(object):
    """Early stop performing
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
    patience : int
        Number of epochs to wait before early stop
        if the metric stops getting improved
    taskname : str or None
        Filename for storing the model checkpoint

    """
    def __init__(self, pretrained_model='Null_early_stop.pth', mode='higher', patience=10, filename=None, task_name="None"):
        if filename is None:
            task_name = task_name
            filename = '{}_early_stop.pth'.format(task_name)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
        self.pretrained_model = pretrained_model

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def nosave_step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)
        # print(self.filename)

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        # model.load_state_dict(torch.load(self.filename)['model_state_dict'])
        # model.load_state_dict(torch.load('./model/' + self.filename, map_location=torch.device('cpu'))['model_state_dict'])
        model.load_state_dict(
            torch.load(os.path.join(settings.SITE_ROOT, 'static') + '/media/core/model/' + self.filename,
                       map_location=torch.device('cpu'))['model_state_dict'])

    def load_pretrained_model(self, model):
        pretrained_parameters = ['gnn_layers.0.graph_conv_layer.weight',
                                 'gnn_layers.0.graph_conv_layer.h_bias',
                                 'gnn_layers.0.graph_conv_layer.loop_weight',
                                 'gnn_layers.0.res_connection.weight',
                                 'gnn_layers.0.res_connection.bias',
                                 'gnn_layers.0.bn_layer.weight',
                                 'gnn_layers.0.bn_layer.bias',
                                 'gnn_layers.0.bn_layer.running_mean',
                                 'gnn_layers.0.bn_layer.running_var',
                                 'gnn_layers.0.bn_layer.num_batches_tracked',
                                 'gnn_layers.1.graph_conv_layer.weight',
                                 'gnn_layers.1.graph_conv_layer.h_bias',
                                 'gnn_layers.1.graph_conv_layer.loop_weight',
                                 'gnn_layers.1.res_connection.weight',
                                 'gnn_layers.1.res_connection.bias',
                                 'gnn_layers.1.bn_layer.weight',
                                 'gnn_layers.1.bn_layer.bias',
                                 'gnn_layers.1.bn_layer.running_mean',
                                 'gnn_layers.1.bn_layer.running_var',
                                 'gnn_layers.1.bn_layer.num_batches_tracked']
        pretrained_model = torch.load(self.pretrained_model, map_location=torch.device('cpu'))
        # pretrained_model = torch.load(self.pretrained_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)

    def load_model_attention(self, model):
        pretrained_parameters = ['gnn_layers.0.graph_conv_layer.weight',
                                 'gnn_layers.0.graph_conv_layer.h_bias',
                                 'gnn_layers.0.graph_conv_layer.loop_weight',
                                 'gnn_layers.0.res_connection.weight',
                                 'gnn_layers.0.res_connection.bias',
                                 'gnn_layers.0.bn_layer.weight',
                                 'gnn_layers.0.bn_layer.bias',
                                 'gnn_layers.0.bn_layer.running_mean',
                                 'gnn_layers.0.bn_layer.running_var',
                                 'gnn_layers.0.bn_layer.num_batches_tracked',
                                 'gnn_layers.1.graph_conv_layer.weight',
                                 'gnn_layers.1.graph_conv_layer.h_bias',
                                 'gnn_layers.1.graph_conv_layer.loop_weight',
                                 'gnn_layers.1.res_connection.weight',
                                 'gnn_layers.1.res_connection.bias',
                                 'gnn_layers.1.bn_layer.weight',
                                 'gnn_layers.1.bn_layer.bias',
                                 'gnn_layers.1.bn_layer.running_mean',
                                 'gnn_layers.1.bn_layer.running_var',
                                 'gnn_layers.1.bn_layer.num_batches_tracked',
                                 'weighted_sum_readout.atom_weighting_specific.0.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.0.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.1.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.1.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.2.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.2.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.3.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.3.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.4.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.4.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.5.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.5.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.6.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.6.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.7.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.7.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.8.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.8.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.9.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.9.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.10.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.10.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.11.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.11.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.12.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.12.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.13.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.13.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.14.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.14.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.15.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.15.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.16.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.16.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.17.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.17.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.18.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.18.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.19.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.19.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.20.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.20.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.21.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.21.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.22.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.22.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.23.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.23.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.24.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.24.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.25.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.25.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.26.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.26.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.27.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.27.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.28.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.28.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.29.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.29.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.30.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.30.0.bias',
                                 'weighted_sum_readout.shared_weighting.0.weight',
                                 'weighted_sum_readout.shared_weighting.0.bias',
                                 ]
        pretrained_model = torch.load(self.filename, map_location=torch.device('cpu'))
        # pretrained_model = torch.load(self.pretrained_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)


