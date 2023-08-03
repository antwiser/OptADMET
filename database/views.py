from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.views import generic
from django.utils import timezone
import pandas as pd
import json
from django.db.models import Q
from django.urls import reverse
from django.shortcuts import redirect
from .models import Experi_global, Experi_local, Experi_MMP, Property, Experi_Sortlist, Experi_Property_Structure
from .models import Expand_global, Expand_local, Expand_MMP, Expand_Sortlist, Expand_Property_Structure
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

property2idx = {'logd': 11, 'logp': 12, 'logs': 13, 'Caco-2': 2, 'pgp-inh': 14, 'pgp-sub': 15, 'f20': 9, 'f30': 10,
                'PPB': 16, 'BBB': 1, 'vdss': 17, 'cyp1a2-inh': 3, 'cyp2c19-inh': 4, 'cyp2c9-inh': 5, 'cyp2d6-inh': 6,
                'cyp2d6-sub': 7, 'cyp3a4-inh': 8, 't12': 33, 'AMES': 18, 'BCF': 19, 'Dili': 20, 'ec': 21, 'ei': 22,
                'fdamdd': 23, 'h-ht2': 24, 'herg': 25, 'igc50': 26, 'nr-ahr': 27, 'nr-ar': 28, 'nr-ar-lbd': 29,
                'respiratory': 30, 'sr-are': 31, 'sr-mmp': 32}


def property(request):
    property = Property.objects.all()
    return render(request, 'database/property/index.html', locals())


def structure(request):
    structure = Property.objects.all()
    return render(request, 'database/structure/index.html', locals())


def HighlightReaction(rxn, highlightAtoms, figsize=[400, 200], kekulize=True):
    def _revised(svg_words):
        """
        """
        svg_words = svg_words.replace(
            'stroke-width:2px', 'stroke-width:1.5px').replace(
            'fonts-size:17px', 'fonts-size:15px').replace(
            'stroke-linecap:butt', 'stroke-linecap:square').replace(
            'fill:#FFFFFF', 'fill:none').replace(
            'svg:', '')
        return svg_words

    drawer = rdMolDraw2D.MolDraw2DSVG(*figsize)
    drawer.DrawReaction(rxn, highlightByReactant=True)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return (svg)


def HighlightAtoms(mol, highlightAtoms, figsize=[400, 200], kekulize=True):
    """This function is used for showing which part of fragment matched the SMARTS by the id of atoms.

    :param mol: The molecule to be visualized
    :type mol: rdkit.Chem.rdchem.Mol
    :param highlightAtoms: The atoms to be highlighted
    :type highlightAtoms: tuple
    :param figsize: The resolution ratio of figure
    :type figsize: list
    :return: a figure with highlighted molecule
    :rtype: IPython.core.display.SVG

    """

    def _revised(svg_words):
        """
        """
        svg_words = svg_words.replace(
            'stroke-width:2px', 'stroke-width:1.5px').replace(
            'fonts-size:17px', 'fonts-size:15px').replace(
            'stroke-linecap:butt', 'stroke-linecap:square').replace(
            'fill:#FFFFFF', 'fill:none').replace(
            'svg:', '')
        return svg_words

    mc = Chem.Mol(mol.ToBinary())

    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(*figsize)
    drawer.DrawMolecule(mc, highlightAtoms=highlightAtoms)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return _revised(svg)


def peroperty_datasource(request):
    try:
        if request.method == 'POST':
            draw = int(request.POST.get('draw'))  # 記錄操作次數
            start = int(request.POST.get('start'))  # 起始位置
            length = int(request.POST.get('length'))  # 每頁長度
            search_key = request.POST.get('search[value]')
            property = request.POST.get('submit_data[property]').split(',') if request.POST.get(
                'submit_data[property]') else ''
            credibility = request.POST.get('submit_data[credibility]').split(',') if request.POST.get(
                'submit_data[credibility]') else ''
            variance = request.POST.get('submit_data[variance]').split(',') if request.POST.get(
                'submit_data[variance]') else ''
            dataset = request.POST.get('dataset')
            if property or credibility or variance:
                if dataset == 'experiment':
                    results_all = Experi_global.objects.none()
                else:
                    results_all = Expand_global.objects.none()
                if property:
                    for item in property:
                        if dataset == 'experiment':
                            result = Experi_global.objects.values('transformation_global_ID', 'transformation',
                                                                  'statistical_significance',
                                                                  'variance', 'transformation').filter(
                                transformation_global_ID__icontains=item.lower())
                        else:
                            result = Expand_global.objects.values('transformation_global_ID', 'transformation',
                                                                  'statistical_significance',
                                                                  'variance', 'transformation').filter(
                                transformation_global_ID__icontains=item.lower())
                        results_all |= result
                else:
                    if dataset == 'experiment':
                        results_all = Experi_global.objects.all().values('transformation_global_ID', 'transformation',
                                                                         'statistical_significance',
                                                                         'variance', 'transformation')
                    else:
                        results_all = Expand_global.objects.all().values('transformation_global_ID', 'transformation',
                                                                         'statistical_significance',
                                                                         'variance', 'transformation')
                if dataset == 'experiment':
                    result_credibility = Experi_global.objects.none()
                else:
                    result_credibility = Expand_global.objects.none()
                if credibility == ['1']:
                    for item in credibility:
                        result = results_all.filter(
                            statistical_significance=item)
                        result_credibility |= result
                else:
                    result_credibility = results_all
                if dataset == 'experiment':
                    result_variance = Experi_global.objects.none()
                else:
                    result_variance = Expand_global.objects.none()
                if variance:
                    if variance == ['3']:
                        result = result_credibility.filter(
                            Q(variance=2))
                    elif variance == ['2']:
                        result = result_credibility.filter(
                            Q(variance=1))
                    else:
                        result = result_credibility.filter(variance=3)
                    # for item in variance:
                    #     result = result_credibility.filter(variance=item)
                    result_variance |= result
                if variance:
                    results = result_variance
                elif credibility:
                    results = result_credibility
                else:
                    results = results_all
                counts = len(results)
                results = results[start: start + length]
                datas = []
                for row in results:
                    res = dict()
                    property, type, id = row['transformation_global_ID'].split(
                        '_')
                    res['property'] = property
                    res['idx'] = id
                    res['statistical_significance'] = row['statistical_significance']
                    rxn = AllChem.ReactionFromSmarts(
                        row['transformation'], useSmiles=True)
                    res['svg'] = HighlightReaction(rxn, highlightAtoms=())
                    res['variance'] = row['variance']
                    datas.append(res)
                response = dict()
                response['draw'] = draw
                response['recordsTotal'] = counts
                response['recordsFiltered'] = counts
                response['data'] = datas
                return HttpResponse(json.dumps(response), content_type='application/json')
            else:
                if dataset == 'experiment':
                    results = Experi_global.objects.values('transformation_global_ID', 'statistical_significance', 'transformation',
                                                           'variance').all().order_by('transformation_global_ID')
                else:
                    results = Expand_global.objects.values('transformation_global_ID', 'statistical_significance', 'transformation',
                                                           'variance').all().order_by('transformation_global_ID')
                counts = len(results)
                results = results[start: start + length]
                datas = []
                for row in results:
                    res = dict()
                    property, type, id = row['transformation_global_ID'].split(
                        '_')
                    res['property'] = property
                    res['idx'] = id
                    res['statistical_significance'] = row['statistical_significance']
                    rxn = AllChem.ReactionFromSmarts(
                        row['transformation'], useSmiles=True)
                    res['svg'] = HighlightReaction(rxn, highlightAtoms=())
                    res['variance'] = row['variance']
                    datas.append(res)
                response = dict()
                response['draw'] = draw
                response['recordsTotal'] = counts
                response['recordsFiltered'] = counts
                response['data'] = datas
                return HttpResponse(json.dumps(response), content_type='application/json')
    except Exception as e:
        return HttpResponse(e.args)


def tranformation_detail(request, idx):
    global_info = Experi_global.objects.get(pk=idx)
    property, type, id = global_info.transformation_global_ID.split('_')
    left_fragment = HighlightAtoms(Chem.MolFromSmiles(
        global_info.left_fragment), highlightAtoms=())
    right_fragment = HighlightAtoms(Chem.MolFromSmiles(
        global_info.right_fragment), highlightAtoms=())
    rxn = AllChem.ReactionFromSmarts(
        global_info.transformation, useSmiles=True)
    reaction = HighlightReaction(rxn, highlightAtoms=())
    return render(request, 'database/property/detail_index.html', {
        'global_info': global_info,
        'property': property,
        'id': id,
        'left_fragment': left_fragment,
        'right_fragment': right_fragment,
        'svg': reaction,
    })


def pre_tranformation_detail(request, idx):
    global_info = Expand_global.objects.get(pk=idx)
    property, type, id = global_info.transformation_global_ID.split('_')
    left_fragment = HighlightAtoms(Chem.MolFromSmiles(
        global_info.left_fragment), highlightAtoms=())
    right_fragment = HighlightAtoms(Chem.MolFromSmiles(
        global_info.right_fragment), highlightAtoms=())
    rxn = AllChem.ReactionFromSmarts(
        global_info.transformation, useSmiles=True)
    reaction = HighlightReaction(rxn, highlightAtoms=())
    return render(request, 'database/property/pre_detail_index.html', {
        'global_info': global_info,
        'property': property,
        'id': id,
        'left_fragment': left_fragment,
        'right_fragment': right_fragment,
        'svg': reaction,
    })


def structure_detail(request, idx):
    structure = Experi_Sortlist.objects.get(pk=idx)
    property, type, id = structure.structure_global_id.split('_')
    left_fragment = HighlightAtoms(Chem.MolFromSmiles(
        structure.left_fragment), highlightAtoms=())
    right_fragment = HighlightAtoms(Chem.MolFromSmiles(
        structure.right_fragment), highlightAtoms=())
    rxn = AllChem.ReactionFromSmarts(structure.transformation, useSmiles=True)
    reaction = HighlightReaction(rxn, highlightAtoms=())
    return render(request, 'database/structure/detail_index.html', {
        'structure_info': structure,
        'id': id,
        'left_fragment': left_fragment,
        'right_fragment': right_fragment,
        'svg': reaction,
    })


def pre_structure_detail(request, idx):
    structure = Expand_Sortlist.objects.get(pk=idx)
    property, type, id = structure.structure_global_id.split('_')
    left_fragment = HighlightAtoms(Chem.MolFromSmiles(structure.left_fragment),
                                   highlightAtoms=()) if Chem.MolFromSmiles(structure.left_fragment) else ''
    right_fragment = HighlightAtoms(Chem.MolFromSmiles(structure.right_fragment),
                                    highlightAtoms=()) if Chem.MolFromSmiles(structure.right_fragment) else ''
    rxn = AllChem.ReactionFromSmarts(structure.transformation, useSmiles=True)
    reaction = HighlightReaction(rxn, highlightAtoms=())
    return render(request, 'database/structure/pre_detail_index.html', {
        'structure_info': structure,
        'id': id,
        'left_fragment': left_fragment,
        'right_fragment': right_fragment,
        'svg': reaction,
    })


def local_datasource(request, idx):
    try:
        if request.method == 'POST':
            draw = int(request.POST.get('draw'))  # 記錄操作次數
            start = int(request.POST.get('start'))  # 起始位置
            length = int(request.POST.get('length'))  # 每頁長度
            search_key = request.POST.get('search[value]')
            results = Experi_local.objects.values('transformation_local_ID', 'env_1', 'env_2', 'env_3', 'variance',
                                                  'count').filter(
                transformation_global_ID=idx).order_by('count')
            counts = len(results)
            results = results[start: start + length]
            datas = []
            for row in results:
                res = dict()
                res['transformation_local_ID'] = row['transformation_local_ID']
                res['env_1'] = row['env_1']
                res['env_2'] = row['env_2']
                res['env_3'] = row['env_3']
                res['variance'] = row['variance']
                res['count'] = row['count']
                datas.append(res)
            response = dict()
            response['draw'] = draw
            response['recordsTotal'] = counts
            response['recordsFiltered'] = counts
            response['data'] = datas
            return HttpResponse(json.dumps(response), content_type='application/json')
    except Exception as e:
        return HttpResponse(e.args)


def pre_local_datasource(request, idx):
    try:
        if request.method == 'POST':
            draw = int(request.POST.get('draw'))  # 記錄操作次數
            start = int(request.POST.get('start'))  # 起始位置
            length = int(request.POST.get('length'))  # 每頁長度
            search_key = request.POST.get('search[value]')
            results = Expand_local.objects.values('transformation_local_ID', 'env_1', 'env_2', 'env_3', 'variance',
                                                  'count').filter(
                transformation_global_ID=idx).order_by('count')
            counts = len(results)
            results = results[start: start + length]
            datas = []
            for row in results:
                res = dict()
                res['transformation_local_ID'] = row['transformation_local_ID']
                res['env_1'] = row['env_1']
                res['env_2'] = row['env_2']
                res['env_3'] = row['env_3']
                res['variance'] = row['variance']
                res['count'] = row['count']
                datas.append(res)
            response = dict()
            response['draw'] = draw
            response['recordsTotal'] = counts
            response['recordsFiltered'] = counts
            response['data'] = datas
            return HttpResponse(json.dumps(response), content_type='application/json')
    except Exception as e:
        return HttpResponse(e.args)


def global_datasource(request, idx):
    try:
        if request.method == 'POST':
            draw = int(request.POST.get('draw'))  # 記錄操作次數
            start = int(request.POST.get('start'))  # 起始位置
            length = int(request.POST.get('length'))  # 每頁長度
            search_key = request.POST.get('search[value]')
            results = Experi_global.objects.values('transformation_global_ID', 'variance', 'statistical_significance',
                                                   'count').filter(structure_global_ID=idx)
            counts = len(results)
            results = results[start: start + length]
            datas = []
            for row in results:
                res = dict()
                res['transformation_global_ID'] = row['transformation_global_ID']
                res['statistical_significance'] = row['statistical_significance']
                res['count'] = row['count']
                res['variance'] = row['variance']
                res['property'] = row['transformation_global_ID'].split('_')[0]
                datas.append(res)
            response = dict()
            response['draw'] = draw
            response['recordsTotal'] = counts
            response['recordsFiltered'] = counts
            response['data'] = datas
            return HttpResponse(json.dumps(response), content_type='application/json')
    except Exception as e:
        return HttpResponse(e.args)


def pre_global_datasource(request, idx):
    try:
        if request.method == 'POST':
            draw = int(request.POST.get('draw'))  # 記錄操作次數
            start = int(request.POST.get('start'))  # 起始位置
            length = int(request.POST.get('length'))  # 每頁長度
            search_key = request.POST.get('search[value]')
            results = Expand_global.objects.values('transformation_global_ID', 'variance', 'statistical_significance',
                                                   'count').filter(structure_global_ID=idx)
            counts = len(results)
            results = results[start: start + length]
            datas = []
            for row in results:
                res = dict()
                res['transformation_global_ID'] = row['transformation_global_ID']
                res['statistical_significance'] = row['statistical_significance']
                res['count'] = row['count']
                res['variance'] = row['variance']
                res['property'] = row['transformation_global_ID'].split('_')[0]
                datas.append(res)
            response = dict()
            response['draw'] = draw
            response['recordsTotal'] = counts
            response['recordsFiltered'] = counts
            response['data'] = datas
            return HttpResponse(json.dumps(response), content_type='application/json')
    except Exception as e:
        return HttpResponse(e.args)


def ldatasource(request, idx):
    try:
        if request.method == 'POST':
            draw = int(request.POST.get('draw'))  # 記錄操作次數
            start = int(request.POST.get('start'))  # 起始位置
            length = int(request.POST.get('length'))  # 每頁長度
            search_key = request.POST.get('search[value]')
            results = Experi_local.objects.values('variance', 'statistical_significance',
                                                  'count', 'structure_local_ID', 'env_1', 'env_2', 'env_3',
                                                  'transformation_local_ID').filter(
                structure_global_ID=idx).order_by('structure_local_ID')
            counts = len(results)
            results = results[start: start + length]
            datas = []
            for row in results:
                res = dict()
                res['transformation_local_id'] = row['transformation_local_ID']
                res['statistical_significance'] = row['statistical_significance']
                res['count'] = row['count']
                res['variance'] = row['variance']
                res['property'] = row['transformation_local_ID'].split('_')[0]
                res['env_1'] = row['env_1']
                res['env_2'] = row['env_2']
                res['env_3'] = row['env_3']
                datas.append(res)
            response = dict()
            response['draw'] = draw
            response['recordsTotal'] = counts
            response['recordsFiltered'] = counts
            response['data'] = datas
            return HttpResponse(json.dumps(response), content_type='application/json')
    except Exception as e:
        return HttpResponse(e.args)


def pre_ldatasource(request, idx):
    try:
        if request.method == 'POST':
            draw = int(request.POST.get('draw'))  # 記錄操作次數
            start = int(request.POST.get('start'))  # 起始位置
            length = int(request.POST.get('length'))  # 每頁長度
            search_key = request.POST.get('search[value]')
            results = Expand_local.objects.values('variance', 'statistical_significance',
                                                  'count', 'structure_local_ID', 'env_1', 'env_2', 'env_3',
                                                  'transformation_local_ID').filter(
                structure_global_ID=idx).order_by('structure_local_ID')
            counts = len(results)
            results = results[start: start + length]
            datas = []
            for row in results:
                res = dict()
                res['transformation_local_id'] = row['transformation_local_ID']
                res['statistical_significance'] = row['statistical_significance']
                res['count'] = row['count']
                res['variance'] = row['variance']
                res['property'] = row['transformation_local_ID'].split('_')[0]
                res['env_1'] = row['env_1']
                res['env_2'] = row['env_2']
                res['env_3'] = row['env_3']
                datas.append(res)
            response = dict()
            response['draw'] = draw
            response['recordsTotal'] = counts
            response['recordsFiltered'] = counts
            response['data'] = datas
            return HttpResponse(json.dumps(response), content_type='application/json')
    except Exception as e:
        return HttpResponse(e.args)


def local_detail(request, idx):
    result = Experi_local.objects.get(pk=idx)
    property, type, id = result.transformation_global_ID.transformation_global_ID.split(
        '_')
    left_fragment = HighlightAtoms(Chem.MolFromSmiles(
        result.left_fragment), highlightAtoms=())
    right_fragment = HighlightAtoms(Chem.MolFromSmiles(
        result.right_fragment), highlightAtoms=())
    rxn = AllChem.ReactionFromSmarts(result.transformation, useSmiles=True)
    reaction = HighlightReaction(rxn, highlightAtoms=())
    return render(request, 'database/property/local_detail_index.html', {
        'local_info': result,
        'property': property,
        'id': id,
        'left_fragment': left_fragment,
        'right_fragment': right_fragment,
        'svg': reaction,
    })


def pre_local_detail(request, idx):
    result = Expand_local.objects.get(pk=idx)
    property, type, id = result.transformation_global_ID.transformation_global_ID.split(
        '_')
    left_fragment = HighlightAtoms(Chem.MolFromSmiles(
        result.left_fragment), highlightAtoms=())
    right_fragment = HighlightAtoms(Chem.MolFromSmiles(
        result.right_fragment), highlightAtoms=())
    rxn = AllChem.ReactionFromSmarts(
        result.transformation_global_ID.transformation, useSmiles=True)
    reaction = HighlightReaction(rxn, highlightAtoms=())
    return render(request, 'database/property/pre_local_detail_index.html', {
        'local_info': result,
        'property': property,
        'id': id,
        'left_fragment': left_fragment,
        'right_fragment': right_fragment,
        'svg': reaction,
    })


def mmp_datasource(request, idx):
    try:
        if request.method == 'POST':
            draw = int(request.POST.get('draw'))  # 記錄操作次數
            start = int(request.POST.get('start'))  # 起始位置
            length = int(request.POST.get('length'))  # 每頁長度
            search_key = request.POST.get('search[value]')
            results = Experi_MMP.objects.values('id', 'change', 'value_l', 'value_r', 'transformation_global_ID',
                                                'molecule_l', 'molecule_r').filter(
                transformation_global_ID=idx).order_by('-change')
            counts = len(results)
            results = results[start: start + length]
            datas = []
            for row in results:
                res = dict()
                res['idx'] = row['id']
                res['change'] = row['change']
                res['value_l'] = row['value_l']
                res['value_r'] = row['value_r']
                res['path_dir'] = row['id'] // 10000
                res['molecule_l'] = HighlightAtoms(Chem.MolFromSmiles(row['molecule_l']),
                                                   highlightAtoms=(), figsize=[300, 300]) if Chem.MolFromSmiles(
                    row['molecule_l']) else ''
                res['molecule_r'] = HighlightAtoms(Chem.MolFromSmiles(row['molecule_r']),
                                                   highlightAtoms=(), figsize=[300, 300]) if Chem.MolFromSmiles(
                    row['molecule_r']) else ''
                datas.append(res)
            response = dict()
            response['draw'] = draw
            response['recordsTotal'] = counts
            response['recordsFiltered'] = counts
            response['data'] = datas
            return HttpResponse(json.dumps(response), content_type='application/json')
    except Exception as e:
        return HttpResponse(e.args)


def pre_mmp_datasource(request, idx):
    try:
        if request.method == 'POST':
            draw = int(request.POST.get('draw'))  # 記錄操作次數
            start = int(request.POST.get('start'))  # 起始位置
            length = int(request.POST.get('length'))  # 每頁長度
            search_key = request.POST.get('search[value]')
            results = Expand_MMP.objects.values('id', 'change', 'value_l', 'value_r', 'transformation_global_ID',
                                                'molecule_l', 'molecule_r').filter(
                transformation_global_ID=idx).order_by('-change')
            counts = len(results)
            results = results[start: start + length]
            datas = []
            for row in results:
                res = dict()
                res['idx'] = row['id']
                res['change'] = row['change']
                res['value_l'] = row['value_l']
                res['value_r'] = row['value_r']
                res['path_dir'] = row['id'] // 10000
                res['molecule_l'] = HighlightAtoms(Chem.MolFromSmiles(row['molecule_l']),
                                                   highlightAtoms=(), figsize=[300, 300]) if Chem.MolFromSmiles(
                    row['molecule_l']) else ''
                res['molecule_r'] = HighlightAtoms(Chem.MolFromSmiles(row['molecule_r']),
                                                   highlightAtoms=(), figsize=[300, 300]) if Chem.MolFromSmiles(
                    row['molecule_r']) else ''
                datas.append(res)
            response = dict()
            response['draw'] = draw
            response['recordsTotal'] = counts
            response['recordsFiltered'] = counts
            response['data'] = datas
            return HttpResponse(json.dumps(response), content_type='application/json')
    except Exception as e:
        return HttpResponse(e.args)


def mmp_ldatasource(request, idx):
    try:
        if request.method == 'POST':
            draw = int(request.POST.get('draw'))  # 記錄操作次數
            start = int(request.POST.get('start'))  # 起始位置
            length = int(request.POST.get('length'))  # 每頁長度
            search_key = request.POST.get('search[value]')
            results = Experi_MMP.objects.values('id', 'change', 'value_l', 'value_r',
                                                'transformation_global_ID', 'molecule_l', 'molecule_r').filter(
                transformation_local_ID=idx).order_by('-change')
            counts = len(results)
            results = results[start: start + length]
            datas = []
            for row in results:
                res = dict()
                res['idx'] = row['id']
                res['change'] = row['change']
                res['value_l'] = row['value_l']
                res['value_r'] = row['value_r']
                res['property'] = row['transformation_global_ID'].split('_')[0]
                res['molecule_l'] = HighlightAtoms(Chem.MolFromSmiles(row['molecule_l']),
                                                   highlightAtoms=(), figsize=[300, 300]) if Chem.MolFromSmiles(
                    row['molecule_l']) else ''
                res['molecule_r'] = HighlightAtoms(Chem.MolFromSmiles(row['molecule_r']),
                                                   highlightAtoms=(), figsize=[300, 300]) if Chem.MolFromSmiles(
                    row['molecule_r']) else ''
                datas.append(res)
            response = dict()
            response['draw'] = draw
            response['recordsTotal'] = counts
            response['recordsFiltered'] = counts
            response['data'] = datas
            return HttpResponse(json.dumps(response), content_type='application/json')
    except Exception as e:
        return HttpResponse(e.args)


def pre_mmp_ldatasource(request, idx):
    try:
        if request.method == 'POST':
            draw = int(request.POST.get('draw'))  # 記錄操作次數
            start = int(request.POST.get('start'))  # 起始位置
            length = int(request.POST.get('length'))  # 每頁長度
            search_key = request.POST.get('search[value]')
            results = Expand_MMP.objects.values('id', 'change', 'value_l', 'value_r',
                                                'transformation_global_ID', 'molecule_l', 'molecule_r').filter(
                transformation_local_ID=idx).order_by('-change')
            counts = len(results)
            results = results[start: start + length]
            datas = []
            for row in results:
                res = dict()
                res['idx'] = row['id']
                res['change'] = row['change']
                res['value_l'] = row['value_l']
                res['value_r'] = row['value_r']
                res['property'] = row['transformation_global_ID'].split('_')[0]
                res['molecule_l'] = HighlightAtoms(Chem.MolFromSmiles(row['molecule_l']),
                                                   highlightAtoms=(), figsize=[300, 300]) if Chem.MolFromSmiles(
                    row['molecule_l']) else ''
                res['molecule_r'] = HighlightAtoms(Chem.MolFromSmiles(row['molecule_r']),
                                                   highlightAtoms=(), figsize=[300, 300]) if Chem.MolFromSmiles(
                    row['molecule_r']) else ''
                datas.append(res)
            response = dict()
            response['draw'] = draw
            response['recordsTotal'] = counts
            response['recordsFiltered'] = counts
            response['data'] = datas
            return HttpResponse(json.dumps(response), content_type='application/json')
    except Exception as e:
        return HttpResponse(e.args)


def structure_datasource(request):
    try:
        if request.method == 'POST':
            draw = int(request.POST.get('draw'))  # 記錄操作次數
            start = int(request.POST.get('start'))  # 起始位置
            length = int(request.POST.get('length'))  # 每頁長度
            search_key = request.POST.get('search[value]')
            property = request.POST.get('submit_data[property]').split(',') if request.POST.get(
                'submit_data[property]') else ''
            property2 = request.POST.get('submit_data[property2]').split(',') if request.POST.get(
                'submit_data[property2]') else ''
            dataset = request.POST.get('dataset')
            property_all = []
            if property or property2:
                if property:
                    property_all += property
                if property2:
                    property_all += property2
                property_all = list(set(property_all))
                property_idx = [property2idx[item] for item in property_all]
                if dataset == 'experiment':
                    results_all = Experi_Sortlist.objects.all()
                    for item in property_idx:
                        property_structure = results_all.values('structure_global_id', 'transformation').filter(
                            experi_property_structure__property_id=item)
                        results_all = property_structure
                    print(len(results_all))
                else:
                    results_all = Expand_Sortlist.objects.all()
                    for item in property_idx:
                        property_structure = results_all.values('structure_global_id', 'transformation').filter(
                            expand_property_structure__property_id=item)
                        results_all = property_structure
                results = results_all
                counts = len(results)
                results = results[start: start + length]
                datas = []
                for row in results:
                    res = dict()
                    property, type, id = row['structure_global_id'].split('_')
                    res['idx'] = id
                    rxn = AllChem.ReactionFromSmarts(
                        row['transformation'], useSmiles=True)
                    res['svg'] = HighlightReaction(rxn, highlightAtoms=())
                    datas.append(res)
                response = dict()
                response['draw'] = draw
                response['recordsTotal'] = counts
                response['recordsFiltered'] = counts
                response['data'] = datas
                return HttpResponse(json.dumps(response), content_type='application/json')
            else:
                if dataset == 'experiment':
                    results = Experi_Sortlist.objects.values(
                        'structure_global_id', 'transformation').all()
                else:
                    results = Expand_Sortlist.objects.values(
                        'structure_global_id', 'transformation').all()
                counts = len(results)
                results = results[start: start + length]
                datas = []
                for _, row in enumerate(results):
                    res = dict()
                    property, type, id = row['structure_global_id'].split('_')
                    res['property'] = property
                    res['idx'] = id
                    res['fade_id'] = _
                    rxn = AllChem.ReactionFromSmarts(
                        row['transformation'], useSmiles=True)
                    res['svg'] = HighlightReaction(rxn, highlightAtoms=())
                    datas.append(res)
                response = dict()
                response['draw'] = draw
                response['recordsTotal'] = counts
                response['recordsFiltered'] = counts
                response['data'] = datas
                return HttpResponse(json.dumps(response), content_type='application/json')
    except Exception as e:
        return HttpResponse(e.args)
