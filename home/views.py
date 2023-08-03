from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole  # display molecular object
from rdkit.Chem import rdChemReactions
from rdkit.Chem import AllChem as Chem
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.views import generic
import json
from django.urls import reverse
from django.shortcuts import redirect
import os
from django.http import FileResponse
import django
from django.conf import settings
from rdkit import Chem
import _pickle as cPickle
import gzip
from database.models import Property, Experi_Sortlist, Expand_Sortlist
import re
import tempfile
import numpy as np
import pandas as pd
import time
import math
from rdkit import RDLogger
from database.views import HighlightReaction
import static.media.core.optAPI as oapi
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from bisect import bisect_left
import hashlib
from .models import ADMETProperty

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    Image,
    TableStyle,
    PageBreak,
)
from reportlab.platypus import Frame, ListFlowable, ListItem, Flowable
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
    _baseFontNameB,
    _baseFontName,
)
from reportlab.lib.pagesizes import A4
from reportlab.rl_config import defaultPageSize
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.colors import Color, yellow, green, red, black, blue
from reportlab.graphics.shapes import Circle
from reportlab.lib.colors import tan, green
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

RDLogger.DisableLog("rdApp.*")  # ignore rdkit warnings


def index(request):
    return render(request, "home/index.html", {})


def tutorials(request):
    return render(request, "home/tutorial_index.html", {})


def contact(request):
    return render(request, "home/contact_index.html", {})


def download(request):
    exp_global_info = pd.read_csv(
        os.path.join(settings.SITE_ROOT, "static")
        + "/media/download/exp_global_info.csv"
    )
    exp_local_info = pd.read_csv(
        os.path.join(settings.SITE_ROOT, "static")
        + "/media/download/exp_local_info.csv"
    )
    pre_global_info = pd.read_csv(
        os.path.join(settings.SITE_ROOT, "static")
        + "/media/download/pre_global_info.csv"
    )
    pre_local_info = pd.read_csv(
        os.path.join(settings.SITE_ROOT, "static")
        + "/media/download/pre_local_info.csv"
    )
    return render(
        request,
        "home/download_index.html",
        {
            "exp_global_info": exp_global_info,
            "exp_local_info": exp_local_info,
            "pre_global_info": pre_global_info,
            "pre_local_info": pre_local_info,
        },
    )


def publication(request):
    return render(request, "home/publication_index.html", {})


def term(request):
    return render(request, "home/term_index.html", {})


def checker(request):
    properties = Property.objects.all()
    return render(request, "checker/index.html", {"properties": properties})


class OptSearch:
    def __init__(self, mol):
        self.mol = Chem.AddHs(mol)

        self.endpointMap = {
            "all": "_sort",
        }

    def _generatePkl(self, data, pattFile):
        model = {}
        cols = ["Left Fragment", "Right Fragment"]

        for col in cols:
            for smart in data[col].values:
                smart = re.sub("\d+\*", "!#1", smart)
                if smart not in model:
                    patt = Chem.MolFromSmarts(smart)
                    patt.UpdatePropertyCache(strict=False)
                    patt_ = Chem.AddHs(patt, addCoords=True)
                    model[smart] = patt_

        out = cPickle.dumps(model, protocol=-1)

        with gzip.open(pattFile, "wb") as f_out:
            f_out.write(out)
        f_out.close()

        return None

    def search(self, endpoint, mode="left", dataset="exp"):
        assert mode in ["left", "right", "all"]

        cols = {
            "left": ["Left Fragment"],
            "right": ["Right Fragment"],
            "all": ["Left Fragment", "Right Fragment"],
        }[mode]

        name = self.endpointMap[endpoint]
        base_path = os.path.join(settings.SITE_ROOT, "static")
        transFile = base_path + "/media/submatch/datas/" + f"{dataset}{name}.csv"
        pattFile = base_path + "/media/submatch/datas/" + f"{dataset}{name}.pkl.gz"

        data = pd.read_csv(transFile)

        if not os.path.exists(pattFile):
            self._generatePkl(data, pattFile)

        pattMap = cPickle.load(gzip.open(pattFile, "rb"))

        def bar(row, cols):
            for col in cols:
                smart = row[col]
                smart = re.sub("\d+\*", "!#1", smart)
                patt = pattMap[smart]

                if self.mol.HasSubstructMatch(patt):
                    return True

            return False

        bo = data.apply(lambda row: bar(row, cols), axis=1)

        out = data[bo]
        return out


property2idx = {
    "logd": 11,
    "logp": 12,
    "logs": 13,
    "Caco-2": 2,
    "pgp-inh": 14,
    "pgp-sub": 15,
    "f20": 9,
    "f30": 10,
    "PPB": 16,
    "BBB": 1,
    "vdss": 17,
    "cyp1a2-inh": 3,
    "cyp2c19-inh": 4,
    "cyp2c9-inh": 5,
    "cyp2d6-inh": 6,
    "cyp2d6-sub": 7,
    "cyp3a4-inh": 8,
    "t12": 33,
    "AMES": 18,
    "BCF": 19,
    "Dili": 20,
    "ec": 21,
    "ei": 22,
    "fdamdd": 23,
    "h-ht2": 24,
    "herg": 25,
    "igc50": 26,
    "nr-ahr": 27,
    "nr-ar": 28,
    "nr-ar-lbd": 29,
    "respiratory": 30,
    "sr-are": 31,
    "sr-mmp": 32,
}

property2header = {
    "logd": "LogD",
    "logp": "LogP",
    "logs": "LogS",
    "Caco-2": "Caco-2",
    "pgp-inh": "Pgp-inh",
    "pgp-sub": "Pgp-sub",
    "f20": "F(20%)",
    "f30": "F(30%)",
    "PPB": "PPB",
    "BBB": "BBB",
    "vdss": "VDss",
    "cyp1a2-inh": "CYP1A2-inh",
    "cyp2c19-inh": "CYP2C19-inh",
    "cyp2c9-inh": "CYP2C9-inh",
    "cyp2d6-inh": "CYP2D6-inh",
    "cyp2d6-sub": "CYP2D6-sub",
    "cyp3a4-inh": "CYP3A4-inh",
    "t12": "T12",
    "AMES": "Ames",
    "BCF": "BCF",
    "Dili": "DILI",
    "ec": "EC",
    "ei": "EI",
    "fdamdd": "FDAMDD",
    "h-ht2": "H-HT",
    "herg": "hERG",
    "igc50": "IGC50",
    "nr-ahr": "NR-AhR",
    "nr-ar": "NR-AR",
    "nr-ar-lbd": "NR-AR-LBD",
    "respiratory": "Respiratory",
    "sr-are": "SR-ARE",
    "sr-mmp": "SR-MMP",
}

idx2property = {
    11: "logd",
    12: "logp",
    13: "logs",
    2: "Caco-2",
    14: "pgp-inh",
    15: "pgp-sub",
    9: "f20",
    10: "f30",
    16: "PPB",
    1: "BBB",
    17: "vdss",
    3: "cyp1a2-inh",
    4: "cyp2c19-inh",
    5: "cyp2c9-inh",
    6: "cyp2d6-inh",
    7: "cyp2d6-sub",
    8: "cyp3a4-inh",
    33: "t12",
    18: "AMES",
    19: "BCF",
    20: "Dili",
    21: "ec",
    22: "ei",
    23: "fdamdd",
    24: "h-ht2",
    25: "herg",
    26: "igc50",
    27: "nr-ahr",
    28: "nr-ar",
    29: "nr-ar-lbd",
    30: "respiratory",
    31: "sr-are",
    32: "sr-mmp",
}

idx2header = {
    11: "LogD",
    12: "LogP",
    13: "LogS",
    2: "Caco-2",
    14: "Pgp-inh",
    15: "Pgp-sub",
    9: "F(20%)",
    10: "F(30%)",
    16: "PPB",
    1: "BBB",
    17: "VDss",
    3: "CYP1A2-inh",
    4: "CYP2C19-inh",
    5: "CYP2C9-inh",
    6: "CYP2D6-inh",
    7: "cyp2d6-sub",
    8: "CYP2D6-sub",
    33: "T12",
    18: "Ames",
    19: "BCF",
    20: "DILI",
    21: "EC",
    22: "EI",
    23: "FDAMDD",
    24: "H-HT",
    25: "hERG",
    26: "IGC50",
    27: "NR-AhR",
    28: "NR-AR",
    29: "NR-AR-LBD",
    30: "Respiratory",
    31: "SR-ARE",
    32: "SR-MMP",
}

idx2propertydisplay = {
    11: "LogD",
    12: "LogP",
    13: "LogS",
    2: "Caco-2",
    14: "pgp-inh",
    15: "pgp-sub",
    9: "f20",
    10: "f30",
    16: "PPB",
    1: "BBB",
    17: "vdss",
    3: "cyp1a2-inh",
    4: "cyp2c19-inh",
    5: "cyp2c9-inh",
    6: "cyp2d6-inh",
    7: "cyp2d6-sub",
    8: "cyp3a4-inh",
    33: "t12",
    18: "AMES",
    19: "BCF",
    20: "Dili",
    21: "ec",
    22: "ei",
    23: "fdamdd",
    24: "h-ht2",
    25: "herg",
    26: "igc50",
    27: "nr-ahr",
    28: "nr-ar",
    29: "nr-ar-lbd",
    30: "respiratory",
    31: "sr-are",
    32: "sr-mmp",
}

def getMD5(request: str) -> str:
    md5_obj = hashlib.md5()
    md5_obj.update(request.encode('utf-8'))
    return md5_obj.hexdigest()


def checkercal(request):
    if request.method == "POST":
        message = ""
        method = request.POST.get("method")
        smiles = ""
        properties = Property.objects.all()
        ori_mol = None
        if method == "1":
            smiles = request.POST.get("smiles")
            # ori_property = request.POST.getlist('property1')
            property1 = request.POST.getlist("property1a")
            property2 = request.POST.getlist("property1b")
            if not smiles:
                message = "Please input molecule!"
                return render(request, "checker/index.html", locals())
            ori_property = property1 + property2 if property2 != [""] else property1
            dataset = request.POST.get("dataset")
            property = [property2idx[item] for item in ori_property]
            if len(property) == 0:
                message = "Please select at least one property!"
                return render(request, "checker/index.html", locals())
            if len(property) > 2:
                message = "Please do not choose more than two properties!"
                return render(request, "checker/index.html", locals())
            ori_mol = Chem.MolFromSmiles(smiles)
            if not ori_mol:  # mol为空，说明输入的SMILES字符串是错误的
                message = "The SMILES is invalid! please check!"
                return render(request, "checker/index.html", locals())
        else:
            # ori_property = request.POST.getlist('property2')
            property1 = request.POST.getlist("property2a")
            property2 = request.POST.getlist("property2b")
            ori_property = property1 + property2 if property2 != [""] else property1
            # print(ori_property)
            property = [property2idx[item] for item in ori_property]
            dataset = request.POST.get("dataset2")
            m = str(request.POST.get("mol", ""))
            ori_mol = Chem.MolFromMolBlock(m)
            if not ori_mol:  # mol为空，说明输入的SMILES字符串是错误的
                message = "The SMILES is invalid! please check!"
                return render(request, "checker/index.html", locals())
            smiles = Chem.MolToSmiles(ori_mol)
            if not smiles:  # mol为空，说明输入的SMILES字符串是错误的
                message = "The SMILES is invalid! please check!"
                return render(request, "checker/index.html", locals())
            ori_mol = Chem.RemoveHs(ori_mol)
            if len(property) == 0:
                message = "Please select at least one property!"
                return render(request, "checker/index.html", locals())
            if len(property) > 2:
                message = "Please do not choose more than two properties!"
                return render(request, "checker/index.html", locals())
        if dataset == "1":
            results_all = Experi_Sortlist.objects.values(
                "transformation",
                "left_fragment",
                "right_fragment",
                "transformation_reaction_SMARTS",
                "structure_global_id",
            ).all()
            for item in property:
                property_structure = results_all.filter(
                    experi_property_structure__property_id=int(item)
                )
                results_all = property_structure
            checked_datas = pd.DataFrame(
                list(results_all),
                columns=[
                    "transformation",
                    "left_fragment",
                    "right_fragment",
                    "transformation_reaction_SMARTS",
                    "structure_global_id",
                ],
            )
        else:
            results_all = Expand_Sortlist.objects.values(
                "transformation",
                "left_fragment",
                "right_fragment",
                "transformation_reaction_SMARTS",
                "structure_global_id",
            ).all()
            for item in property:
                property_structure = results_all.filter(
                    expand_property_structure__property_id=int(item)
                )
                results_all = property_structure
            checked_datas = pd.DataFrame(
                list(results_all),
                columns=[
                    "transformation",
                    "left_fragment",
                    "right_fragment",
                    "transformation_reaction_SMARTS",
                    "structure_global_id",
                ],
            )
        # checked_datas.to_csv('check.csv', index=False)
        if dataset == "1":
            out = get_transform(
                ori_mol,
                checked_datas,
                dataset="exp",
                left="left_fragment",
                right="right_fragment",
                trans="transformation_reaction_SMARTS",
            )
        else:
            out = get_transform(
                ori_mol,
                checked_datas,
                dataset="pre",
                left="left_fragment",
                right="right_fragment",
                trans="transformation_reaction_SMARTS",
            )
        # out.to_csv('test_trans.csv', index=False)
        if len(out) == 0:
            message = "Failure to find an optimized molecule for this molecule!"
            return render(request, "checker/index.html", {"message": message})
            # 这里要写  如果没有找到规则的相应逻辑
        length = out.newSmiles.map(lambda x: len(x))
        mol_datas = pd.DataFrame(
            columns=[
                "structure_id",
                "smiles",
                "transformation",
                "left_fragment",
                "right_fragment",
                "flag",
            ]
        )
        mol_datas = mol_datas.append(
            {
                "structure_id": "origin",
                "smiles": smiles,
                "transformation": "",
                "left_fragment": "",
                "right_fragment": "",
                "flag": "",
            },
            ignore_index=True,
        )
        for index, row in out.iterrows():
            molecules = row["newSmiles"]
            for molecule in molecules:
                mol_datas = mol_datas.append(
                    {
                        "structure_id": row["structure_global_id"],
                        "smiles": molecule.strip(),
                        "transformation": row["transformation"],
                        "left_fragment": row["left_fragment"],
                        "right_fragment": row["right_fragment"],
                        "flag": row["flag"],
                    },
                    ignore_index=True,
                )
        print("规则数:", len(out))
        print("分子数:", len(mol_datas) - 1)
        print(mol_datas)
        # mol_datas.to_csv('test.csv', index=False)
        # 预测
        mol_datas = mol_datas[mol_datas['smiles'].map(lambda x: '[3*]' not in x)]
        add_datas = mol_datas[mol_datas['transformation'].map(lambda x: x == '[1*]O>>[1*]N(C)C')]
        mol_datas = mol_datas.sample(n=201) if mol_datas.shape[0] > 201 else mol_datas
        mol_datas = pd.concat([mol_datas, add_datas], axis=0)
        print(len(mol_datas))
        tmpf = tempfile.NamedTemporaryFile()
        file_name = tmpf.name.split("/")[-1]
        t = int(time.time())
        file_name = file_name + str(t)
        cache_bin_path = (
            os.path.join(settings.SITE_ROOT, "static")
            + "/files/cache/tmp/"
            + file_name
            + ".bin"
        )
        cache_csv_path = (
            os.path.join(settings.SITE_ROOT, "static")
            + "/files/cache/tmp/"
            + file_name
            + ".csv"
        )
        result_path = (
            os.path.join(settings.SITE_ROOT, "static")
            + "/files/result/tmp/"
            + file_name
            + ".csv"
        )
        meta_path = (
            os.path.join(settings.SITE_ROOT, "static")
            + "/files/meta/"
            + file_name
            + ".json"
        )
        # 查看在数据库中是否存在
        # mol_datas['smiles_md5'] = mol_datas['smiles'].map(lambda x: getMD5(x))
        # mol_datas['exist_flag'] = mol_datas['smiles_md5'].map(lambda x: bool(ADMETProperty.objects.filter(md5=x)))
        returncode, success_number, result_df = oapi.multiSMILES(
            mol_datas, cache_bin_path, cache_csv_path, result_path
        )
        origin_molecule = result_df.iloc[0]
        result_df = result_df.iloc[1:]
        ori_idx_dict = dict()
        ori_data_dict = dict()
        data_dict = dict()
        for item in ori_property:
            item = property2header[item]
            max = np.ceil(result_df[item].max())
            min = np.floor(result_df[item].min())
            space = (max - min) / 5
            x_axis = [round(min + i * space, 1) for i in range(6)]
            ori_index = bisect_left(x_axis, origin_molecule[item])
            ori_index = ori_index if ori_index <= 5 else 5
            ori_idx_dict[item] = ori_index
            ori_data_dict[item] = float(origin_molecule[item])
            fenzu = pd.cut(result_df[item].values, x_axis, right=False)
            value_count = fenzu.value_counts()
            value_count.index = value_count.index.astype(str)
            value_count_dict = value_count.to_dict()
            data_dict[item] = value_count_dict
        meta_info = dict()
        meta_info["property"] = "_".join("%s" % item for item in property)
        meta_info["filename"] = str(file_name.split("/")[-1])
        meta_info["dataset"] = dataset
        meta_info["smiles"] = smiles
        meta_info["ori_idx"] = ori_idx_dict
        meta_info["data"] = data_dict
        meta_info["ori_data"] = ori_data_dict
        json_str = json.dumps(meta_info)
        with open(meta_path, "w") as json_file:
            json_file.write(json_str)
        print("这里开始跳转")
        return HttpResponseRedirect(
            reverse("home:result", kwargs={"filename": str(file_name.split("/")[-1])})
        )
    else:
        return render(request, "checker/index.html", locals())


def final_result_file(request, filename, properties):
    meta_path = (
        os.path.join(settings.SITE_ROOT, "static") + "/files/meta/" + filename + ".json"
    )
    with open(meta_path, "r", encoding="UTF-8") as f:
        meta_info = json.load(f)
    result_datas = pd.read_csv(
        os.path.join(settings.SITE_ROOT, "static")
        + "/files/result/tmp/"
        + filename
        + ".csv"
    )
    ori_moleucle_data = result_datas.iloc[0]
    dataset = meta_info["dataset"]
    property = meta_info["property"]
    smiles = meta_info["smiles"]
    properties = properties.split("_") + property.split("_")
    properties = list(set(properties))
    new_property = "_".join(properties)
    properties = [property2header[idx2property[int(item)]] for item in properties]
    structure = HighlightAtoms(
        Chem.MolFromSmiles(smiles), highlightAtoms=(), figsize=[200, 200]
    )
    legend = list(meta_info["data"])
    ori_idx = meta_info["ori_idx"]
    ori_data = dict()
    for item in properties:
        ori_data[item] = ori_moleucle_data[item]
    # ori_data = meta_info['ori_data']
    download_path = (
        "/deploy/optadmet/static/files/final_result/tmp/" + filename + ".csv"
    )
    return render(
        request,
        "checker/final_result_index.html",
        {
            "property": new_property,
            "filename": filename,
            "dataset": dataset,
            "smiles": smiles,
            "structure": structure,
            "data": meta_info["data"],
            "legend": legend,
            "ori_idx": ori_idx,
            "ori_data": ori_data,
            "download_path": download_path,
            "properties": properties,
        },
    )


def staticity(generated_mols):
    result = dict()
    length = len(generated_mols)

    s = ((generated_mols["LogS"] <= 0.5) & (generated_mols["LogS"] >= -4)).sum()
    w = (
        ((generated_mols["LogS"] > 0.5) & (generated_mols["LogS"] < 1))
        | ((generated_mols["LogS"] > -5) & (generated_mols["LogS"] < -4))
    ).sum()
    result["logs"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["LogD"] <= 3) & (generated_mols["LogD"] >= 0)).sum()
    w = (
        ((generated_mols["LogD"] > -1) & (generated_mols["LogD"] < 0))
        | ((generated_mols["LogD"] > 3) & (generated_mols["LogD"] < 4))
    ).sum()
    result["logd"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["LogP"] <= 3) & (generated_mols["LogP"] >= 0)).sum()
    w = (
        ((generated_mols["LogP"] > 0) & (generated_mols["LogP"] < 1))
        | ((generated_mols["LogD"] > 3) & (generated_mols["LogP"] < 4))
    ).sum()
    result["logp"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["Caco-2"] >= -5.15).sum()
    w = ((generated_mols["Caco-2"] > -7) & (generated_mols["Caco-2"] < -5.15)).sum()
    result["Caco2"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["MDCK"] > 0.000002).sum()
    w = (
        (generated_mols["MDCK"] > 0.000001) & (generated_mols["MDCK"] < 0.000002)
    ).sum()
    result["MDCK"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["Pgp-inh"] <= 0.3) & (generated_mols["Pgp-inh"] >= 0)).sum()
    w = ((generated_mols["Pgp-inh"] <= 0.7) & (generated_mols["Pgp-inh"] > 0.3)).sum()
    result["pgpinh"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["Pgp-sub"] <= 0.3) & (generated_mols["Pgp-sub"] >= 0)).sum()
    w = ((generated_mols["Pgp-sub"] <= 0.7) & (generated_mols["Pgp-sub"] > 0.3)).sum()
    result["pgpsub"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["HIA"] <= 0.3) & (generated_mols["HIA"] >= 0)).sum()
    w = ((generated_mols["HIA"] <= 0.7) & (generated_mols["HIA"] > 0.3)).sum()
    result["HIA"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["F(20%)"] <= 0.3) & (generated_mols["F(20%)"] >= 0)).sum()
    w = ((generated_mols["F(20%)"] <= 0.7) & (generated_mols["F(20%)"] > 0.3)).sum()
    result["f20"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["F(30%)"] <= 0.3) & (generated_mols["F(30%)"] >= 0)).sum()
    w = ((generated_mols["F(30%)"] <= 0.7) & (generated_mols["F(30%)"] > 0.3)).sum()
    result["f30"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["PPB"] <= 90).sum()
    w = ((generated_mols["PPB"] < 95) & (generated_mols["PPB"] > 90)).sum()
    result["PPB"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["BBB"] <= 0.3) & (generated_mols["BBB"] >= 0)).sum()
    w = ((generated_mols["BBB"] <= 0.7) & (generated_mols["BBB"] > 0.3)).sum()
    result["BBB"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["VDss"] <= 20) & (generated_mols["VDss"] >= 0.04)).sum()
    w = (
        ((generated_mols["VDss"] > 0.02) & (generated_mols["VDss"] < 0.04))
        | ((generated_mols["VDss"] > 20) & (generated_mols["VDss"] < 50))
    ).sum()
    result["vdss"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["Fu"] >= 5).sum()
    w = ((generated_mols["Fu"] < 5) & (generated_mols["Fu"] > 4)).sum()
    result["Fu"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["CYP1A2-inh"] <= 0.4).sum()
    w = (
        (generated_mols["CYP1A2-inh"] <= 0.7) & (generated_mols["CYP1A2-inh"] > 0.4)
    ).sum()
    result["cyp1a2inh"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["CYP1A2-sub"] <= 0.4).sum()
    w = (
        (generated_mols["CYP1A2-sub"] <= 0.7) & (generated_mols["CYP1A2-sub"] > 0.4)
    ).sum()
    result["cyp1a2sub"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["CYP2C19-inh"] <= 0.4).sum()
    w = (
        (generated_mols["CYP2C19-inh"] <= 0.7) & (generated_mols["CYP2C19-inh"] > 0.4)
    ).sum()
    result["cyp2c19inh"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["CYP2C19-sub"] <= 0.4).sum()
    w = (
        (generated_mols["CYP2C19-sub"] <= 0.7) & (generated_mols["CYP2C19-sub"] > 0.4)
    ).sum()
    result["cyp2c19sub"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["CYP2C9-inh"] <= 0.4).sum()
    w = (
        (generated_mols["CYP2C9-inh"] <= 0.7) & (generated_mols["CYP2C9-inh"] > 0.4)
    ).sum()
    result["cyp2c9inh"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["CYP2C9-sub"] <= 0.4).sum()
    w = (
        (generated_mols["CYP2C9-sub"] <= 0.7) & (generated_mols["CYP2C9-sub"] > 0.4)
    ).sum()
    result["cyp2c9sub"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["CYP2D6-inh"] <= 0.4).sum()
    w = (
        (generated_mols["CYP2D6-inh"] <= 0.7) & (generated_mols["CYP2D6-inh"] > 0.4)
    ).sum()
    result["cyp2d6inh"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["CYP2D6-sub"] <= 0.4).sum()
    w = (
        (generated_mols["CYP2D6-sub"] <= 0.7) & (generated_mols["CYP2D6-sub"] > 0.4)
    ).sum()
    result["cyp2d6sub"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["CYP3A4-inh"] <= 0.4).sum()
    w = (
        (generated_mols["CYP3A4-inh"] <= 0.7) & (generated_mols["CYP3A4-inh"] > 0.4)
    ).sum()
    result["cyp3a4inh"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["CYP3A4-sub"] <= 0.4).sum()
    w = (
        (generated_mols["CYP3A4-sub"] <= 0.7) & (generated_mols["CYP3A4-sub"] > 0.4)
    ).sum()
    result["cyp3a4sub"] = {0: s, 1: w, 2: length - s - w}

    s = (generated_mols["CL"] >= 5).sum()
    w = ((generated_mols["CL"] < 5) & (generated_mols["CL"] > 4)).sum()
    result["CL"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["T12"] <= 0.3) & (generated_mols["T12"] >= 0)).sum()
    w = ((generated_mols["T12"] <= 0.7) & (generated_mols["T12"] > 0.3)).sum()
    result["t12"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["Ames"] <= 0.3) & (generated_mols["Ames"] >= 0)).sum()
    w = ((generated_mols["Ames"] <= 0.7) & (generated_mols["Ames"] > 0.3)).sum()
    result["AMES"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["ROA"] <= 0.3) & (generated_mols["ROA"] >= 0)).sum()
    w = ((generated_mols["ROA"] <= 0.7) & (generated_mols["ROA"] > 0.3)).sum()
    result["ROA"] = {0: s, 1: w, 2: length - s - w}

    s = (
        (generated_mols["Respiratory"] <= 0.3) & (generated_mols["Respiratory"] >= 0)
    ).sum()
    w = (
        (generated_mols["Respiratory"] <= 0.7) & (generated_mols["Respiratory"] > 0.3)
    ).sum()
    result["Respiratory"] = {0: s, 1: w, 2: length - s - w}

    # s = ((generated_mols['BCF'] <= 0.3) & (generated_mols['BCF'] >= 0)).sum()
    # w = ((generated_mols['BCF'] <= 0.7) & (generated_mols['BCF'] > 0.3)).sum()
    # result['BCF'] = {0: s, 1: w, 2: length - s - w}

    s = (
        (generated_mols["Carcinogenicity"] <= 0.3)
        & (generated_mols["Carcinogenicity"] >= 0)
    ).sum()
    w = (
        (generated_mols["Carcinogenicity"] <= 0.7)
        & (generated_mols["Carcinogenicity"] > 0.3)
    ).sum()
    result["Carcinogenicity"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["SkinSen"] <= 0.3) & (generated_mols["SkinSen"] >= 0)).sum()
    w = ((generated_mols["SkinSen"] <= 0.7) & (generated_mols["SkinSen"] > 0.3)).sum()
    result["SkinSen"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["DILI"] <= 0.3) & (generated_mols["DILI"] >= 0)).sum()
    w = ((generated_mols["DILI"] <= 0.7) & (generated_mols["DILI"] > 0.3)).sum()
    result["Dili"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["EC"] <= 0.3) & (generated_mols["EC"] >= 0)).sum()
    w = ((generated_mols["EC"] <= 0.7) & (generated_mols["EC"] > 0.3)).sum()
    result["ec"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["EI"] <= 0.3) & (generated_mols["EI"] >= 0)).sum()
    w = ((generated_mols["EI"] <= 0.7) & (generated_mols["EI"] > 0.3)).sum()
    result["ei"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["FDAMDD"] <= 0.3) & (generated_mols["FDAMDD"] >= 0)).sum()
    w = ((generated_mols["FDAMDD"] <= 0.7) & (generated_mols["FDAMDD"] > 0.3)).sum()
    result["fdamdd"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["H-HT"] <= 0.3) & (generated_mols["H-HT"] >= 0)).sum()
    w = ((generated_mols["H-HT"] <= 0.7) & (generated_mols["H-HT"] > 0.3)).sum()
    result["hht2"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["hERG"] <= 0.3) & (generated_mols["hERG"] >= 0)).sum()
    w = ((generated_mols["hERG"] <= 0.7) & (generated_mols["hERG"] > 0.3)).sum()
    result["herg"] = {0: s, 1: w, 2: length - s - w}

    # s = ((generated_mols['IGC50'] <= 0.3) & (generated_mols['IGC50'] >= 0)).sum()
    # w = ((generated_mols['IGC50'] <= 0.7) & (generated_mols['IGC50'] > 0.3)).sum()
    # result['igc50'] = {0: s, 1: w, 2: length - s - w}

    # s = (generated_mols['LC50'] <= 0.5).sum()
    # result['lc50'] = {0: s, 1: 0, 2: length - s}

    # s = (generated_mols['LC50DM'] <= 0.5).sum()
    # result['lc50dm'] = {0: s, 1: 0, 2: length - s}

    s = ((generated_mols["NR-AhR"] <= 0.3) & (generated_mols["NR-AhR"] >= 0)).sum()
    w = ((generated_mols["NR-AhR"] <= 0.7) & (generated_mols["NR-AhR"] > 0.3)).sum()
    result["nrahr"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["NR-AR"] <= 0.3) & (generated_mols["NR-AR"] >= 0)).sum()
    w = ((generated_mols["NR-AR"] <= 0.7) & (generated_mols["NR-AR"] > 0.3)).sum()
    result["nrar"] = {0: s, 1: w, 2: length - s - w}

    s = (
        (generated_mols["NR-AR-LBD"] <= 0.3) & (generated_mols["NR-AR-LBD"] >= 0)
    ).sum()
    w = (
        (generated_mols["NR-AR-LBD"] <= 0.7) & (generated_mols["NR-AR-LBD"] > 0.3)
    ).sum()
    result["nrarlbd"] = {0: s, 1: w, 2: length - s - w}

    s = (
        (generated_mols["NR-Aromatase"] <= 0.3) & (generated_mols["NR-Aromatase"] >= 0)
    ).sum()
    w = (
        (generated_mols["NR-Aromatase"] <= 0.7) & (generated_mols["NR-Aromatase"] > 0.3)
    ).sum()
    result["nraromatase"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["NR-ER"] <= 0.3) & (generated_mols["NR-ER"] >= 0)).sum()
    w = ((generated_mols["NR-ER"] <= 0.7) & (generated_mols["NR-ER"] > 0.3)).sum()
    result["nrer"] = {0: s, 1: w, 2: length - s - w}

    s = (
        (generated_mols["NR-ER-LBD"] <= 0.3) & (generated_mols["NR-ER-LBD"] >= 0)
    ).sum()
    w = (
        (generated_mols["NR-ER-LBD"] <= 0.7) & (generated_mols["NR-ER-LBD"] > 0.3)
    ).sum()
    result["nrerlbd"] = {0: s, 1: w, 2: length - s - w}

    s = (
        (generated_mols["NR-PPAR-gamma"] <= 0.3)
        & (generated_mols["NR-PPAR-gamma"] >= 0)
    ).sum()
    w = (
        (generated_mols["NR-PPAR-gamma"] <= 0.7)
        & (generated_mols["NR-PPAR-gamma"] > 0.3)
    ).sum()
    result["nrppargamma"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["SR-ARE"] <= 0.3) & (generated_mols["SR-ARE"] >= 0)).sum()
    w = ((generated_mols["SR-ARE"] <= 0.7) & (generated_mols["SR-ARE"] > 0.3)).sum()
    result["srare"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["SR-ATAD5"] <= 0.3) & (generated_mols["SR-ATAD5"] >= 0)).sum()
    w = ((generated_mols["SR-ATAD5"] <= 0.7) & (generated_mols["SR-ATAD5"] > 0.3)).sum()
    result["sratad5"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["SR-HSE"] <= 0.3) & (generated_mols["SR-HSE"] >= 0)).sum()
    w = ((generated_mols["SR-HSE"] <= 0.7) & (generated_mols["SR-HSE"] > 0.3)).sum()
    result["srhse"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["SR-MMP"] <= 0.3) & (generated_mols["SR-MMP"] >= 0)).sum()
    w = ((generated_mols["SR-MMP"] <= 0.7) & (generated_mols["SR-MMP"] > 0.3)).sum()
    result["srmmp"] = {0: s, 1: w, 2: length - s - w}

    s = ((generated_mols["SR-p53"] <= 0.3) & (generated_mols["SR-p53"] >= 0)).sum()
    w = ((generated_mols["SR-p53"] <= 0.7) & (generated_mols["SR-p53"] > 0.3)).sum()
    result["srp53"] = {0: s, 1: w, 2: length - s - w}

    return result


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NpEncoder, self).default(obj)


def result_file(request, filename):
    meta_path = (
        os.path.join(settings.SITE_ROOT, "static") + "/files/meta/" + filename + ".json"
    )
    with open(meta_path, "r", encoding="UTF-8") as f:
        meta_info = json.load(f)
    dataset = meta_info["dataset"]
    property = meta_info["property"]
    smiles = meta_info["smiles"]
    structure = HighlightAtoms(
        Chem.MolFromSmiles(smiles), highlightAtoms=(), figsize=[320, 320]
    )
    legend = list(meta_info["data"])
    ori_idx = meta_info["ori_idx"]
    ori_data = meta_info["ori_data"]
    download_path = "/deploy/optadmet/static/files/result/tmp/" + filename + ".csv"
    property_name = property.split("_")
    property_name = [idx2propertydisplay[int(item)] for item in property_name]
    all_mols = pd.read_csv(
        os.path.join(settings.SITE_ROOT, "static")
        + "/files/result/tmp/"
        + filename
        + ".csv"
    )
    generated_mols = all_mols.iloc[1:]
    original_mol = all_mols.iloc[0]
    result = dict()
    origin_result = dict()
    for item in header2dec.keys():
        origin_result[header2dec[item]] = value2decision[header2dec[item]](
            original_mol[item]
        )
    result["origin"] = origin_result
    static_info = staticity(generated_mols)
    result["generated"] = static_info
    json_path = (
        os.path.join(settings.SITE_ROOT, "static")
        + "/files/summary_info/"
        + filename
        + ".json"
    )
    if not os.path.exists(json_path):
        with open(json_path, "w") as f:
            f.write(json.dumps(result, cls=NpEncoder))

    return render(
        request,
        "checker/result_index.html",
        {
            "property": property,
            "filename": filename,
            "dataset": dataset,
            "smiles": smiles,
            "structure": structure,
            "data": meta_info["data"],
            "legend": legend,
            "ori_idx": ori_idx,
            "ori_data": ori_data,
            "download_path": download_path,
            "property_name": property_name,
            "info": static_info,
        },
    )


def result_mol(request, filename, index, data_idx):
    result_path = (
        os.path.join(settings.SITE_ROOT, "static")
        + "/files/result/tmp/"
        + filename
        + ".csv"
    )
    datas = pd.read_csv(result_path)
    molecule = datas.iloc[int(index)]
    ori_molecule = datas.iloc[0]
    structure = HighlightAtoms(
        Chem.MolFromSmiles(molecule["smiles"]), highlightAtoms=()
    )
    structure_ori = HighlightAtoms(
        Chem.MolFromSmiles(ori_molecule["smiles"]), highlightAtoms=()
    )
    # rxn = AllChem.ReactionFromSmarts(molecule["transformation"], useSmiles=True)
    # transformation = HighlightReaction(rxn, highlightAtoms=(), figsize=[300, 150])
    direct = molecule['flag']
    if direct == 'left':
        rxn = AllChem.ReactionFromSmarts(molecule["transformation"], useSmiles=True)
        transformation = HighlightReaction(rxn, highlightAtoms=(), figsize=[300, 150])
        trans_expre = molecule["transformation"]
        direction = "forword"
    else:
        smarts = ">>".join(molecule["transformation"].split(">>")[::-1])
        rxn = AllChem.ReactionFromSmarts(smarts, useSmiles=True)
        transformation = HighlightReaction(rxn, highlightAtoms=(), figsize=[300, 150])
        trans_expre = smarts
        direction = "backword"
    return render(
        request,
        "checker/molecule_detail_index.html",
        {
            "data": molecule,
            "ori_data": ori_molecule,
            "structure": structure,
            "filename": filename,
            "index": index,
            "structure_ori": structure_ori,
            "transformation": transformation,
            "data_idx": data_idx,
            "trans_expre": trans_expre,
            "direction": direction,
        },
    )


def result_mol_direct(request, filename, index, data_idx, dire_idx):
    result_path = (
        os.path.join(settings.SITE_ROOT, "static")
        + "/files/result/tmp/"
        + filename
        + ".csv"
    )
    datas = pd.read_csv(result_path)
    molecule = datas.iloc[int(index)]
    ori_molecule = datas.iloc[0]
    structure = HighlightAtoms(
        Chem.MolFromSmiles(molecule["smiles"]), highlightAtoms=()
    )
    structure_ori = HighlightAtoms(
        Chem.MolFromSmiles(ori_molecule["smiles"]), highlightAtoms=()
    )
    direct = molecule['flag']
    if direct == 'left':
        rxn = AllChem.ReactionFromSmarts(molecule["transformation"], useSmiles=True)
        transformation = HighlightReaction(rxn, highlightAtoms=(), figsize=[300, 150])
    else:
        smarts = '>>'.join([i for i in molecule["transformation"].split('>>').reverse()])
        rxn = AllChem.ReactionFromSmarts(molecule["transformation"], useSmiles=True)
        transformation = HighlightReaction(rxn, highlightAtoms=(), figsize=[300, 150])
    return render(
        request,
        "checker/molecule_detail_index.html",
        {
            "data": molecule,
            "ori_data": ori_molecule,
            "structure": structure,
            "filename": filename,
            "index": index,
            "structure_ori": structure_ori,
            "transformation": transformation,
            "data_idx": data_idx,
        },
    )



def get_Para(text, style):
    if type(text) == str:
        return Paragraph(text, style)
    elif type(text) == list:
        return [Paragraph(str(item), style) for item in text]


PAGE_HEIGHT = defaultPageSize[1]
PAGE_WIDTH = defaultPageSize[0]

A4_width, A4_height = A4

styles = getSampleStyleSheet()
pageinfo = ""
header_style = ParagraphStyle(
    name="erase_semantic",
    fontSize=11,
    fontName=_baseFontNameB,
)

header_center_style = ParagraphStyle(
    name="erase_semantic",
    fontSize=11,
    fontName=_baseFontNameB,
    alignment=TA_CENTER,
)

decision_style = ParagraphStyle(
    name="decision_style",
    fontName=_baseFontName,
    fontSize=10,
    leading=12,
    spaceBefore=6,
    alignment=TA_CENTER,
)

physicochemical_comment = get_Para(
    [
        "Log of the aqueous solubility. Optimal: -4~0.5 log mol/L",
        "Log of the octanol/water partition coefficient. Optimal: 0~3",
        "logP at physiological pH 7.4. Optimal: 1~3",
    ],
    style=styles["BodyText"],
)
physicochemical_property = get_Para(["logS", "logP", "logD"], style=styles["BodyText"])

absorption_comment = get_Para(
    [
        "Optimal: higher than -5.15 Log unit",
        "▪ low permeability: < 2 × 10<super>−6</super> cm/s<br/>▪ medium permeability: 2–20 × 10<super>−6</super> cm/s<br/>▪ high passive permeability: > 20 × 10<super>−6</super> cm/s",
        "▪ Category 1: Inhibitor; Category 0: Non-inhibitor;<br/>▪ The output value is the probability of being Pgp-inhibitor",
        "▪ Category 1: substrate; Category 0: Non-substrate;<br/>▪ The output value is the probability of being Pgp-substrate",
        "▪ Human Intestinal Absorption<br/>▪ Category 1: HIA+( HIA < 30%); Category 0: HIA-( HIA < 30%); The output value is the probability of being HIA+",
        "▪ 20% Bioavailability<br/>▪ Category 1: F<sub>20%</sub>+ (bioavailability < 20%); Category 0: F<sub>20%</sub>- (bioavailability ≥ 20%); The output value is the probability of being F<sub>20%</sub>+",
        "▪ 30% Bioavailability<br/>▪ Category 1: F<sub>30%</sub>+ (bioavailability < 30%); Category 0: F<sub>30%</sub>- (bioavailability ≥ 30%); The output value is the probability of being F<sub>30%</sub>+",
    ],
    style=styles["BodyText"],
)
absorption_property = get_Para(
    [
        "Caco-2 Permeability",
        "MDCK Permeability",
        "Pgp-inhibitor",
        "Pgp-substrate",
        "HIA",
        "F<sub>20%</sub>",
        "F<sub>30%</sub>",
    ],
    style=styles["BodyText"],
)

distribution_comment = get_Para(
    [
        "▪ Plasma Protein Binding<br/>▪ Optimal: < 90%. Drugs with high protein-bound may have a low therapeutic index.",
        "▪ Volume Distribution<br/>▪ Optimal: 0.04-20L/kg",
        "▪ Blood-Brain Barrier Penetration<br/>▪ Category 1: BBB+; Category 0: BBB-; The output value is the probability of being BBB+",
        "▪ The fraction unbound in plasms<br/>▪ Low: <5%; Middle: 5~20%; High: > 20%",
    ],
    style=styles["BodyText"],
)
distribution_property = get_Para(
    ["PPB", "VD", "BBB Penetration", "Fu"], style=styles["BodyText"]
)

metabolism_comment = get_Para(
    [
        "▪ Category 1: Inhibitor; Category 0: Non-inhibitor;<br/>▪ The output value is the probability of being inhibitor.",
        "▪ Category 1: Substrate; Category 0: Non-substrate;<br/>▪ The output value is the probability of being substrate.",
        "▪ Category 1: Inhibitor; Category 0: Non-inhibitor;<br/>▪ The output value is the probability of being inhibitor.",
        "▪ Category 1: Substrate; Category 0: Non-substrate;<br/>▪ The output value is the probability of being substrate.",
        "▪ Category 1: Inhibitor; Category 0: Non-inhibitor;<br/>▪ The output value is the probability of being inhibitor.",
        "▪ Category 1: Substrate; Category 0: Non-substrate;<br/>▪ The output value is the probability of being substrate.",
        "▪ Category 1: Inhibitor; Category 0: Non-inhibitor;<br/>▪ The output value is the probability of being inhibitor.",
        "▪ Category 1: Substrate; Category 0: Non-substrate;<br/>▪ The output value is the probability of being substrate.",
        "▪ Category 1: Inhibitor; Category 0: Non-inhibitor;<br/>▪ The output value is the probability of being inhibitor.",
        "▪ Category 1: Substrate; Category 0: Non-substrate;<br/>▪ The output value is the probability of being substrate.",
    ],
    style=styles["BodyText"],
)
metabolism_property = get_Para(
    [
        "CYP1A2 inhibitor",
        "CYP1A2 substrate",
        "CYP2C19 inhibitor",
        "CYP2C19 substrate",
        "CYP2C9 inhibitor",
        "CYP2C9 substrate",
        "CYP2D6 inhibitor",
        "CYP2D6 substrate",
        "CYP3A4 inhibitor",
        "CYP3A4 substrate",
    ],
    style=styles["BodyText"],
)

excretion_comment = get_Para(
    [
        "▪ Clearance<br/>▪ High: >15 mL/min/kg; moderate: 5-15 mL/min/kg; low: <5 mL/min/kg",
        "▪ Category 1: long half-life ; Category 0: short half-life;<br/>▪ long half-life: >3h; short half-life: <3h<br/>▪ The output value is the probability of having long half-life.",
    ],
    style=styles["BodyText"],
)
excretion_property = get_Para(["CL", "T<sub>1/2</sub>"], style=styles["BodyText"])

toxicity_comment = get_Para(
    [
        "▪ Category 1: active; Category 0: inactive;<br/>▪ The output value is the probability of being active.",
        "▪ Human Hepatotoxicity<br/>▪ Category 1: H-HT positive(+); Category 0: H-HT negative(-);<br/>▪ The output value is the probability of being toxic.",
        "▪ Drug Induced Liver Injury.<br/>▪ Category 1: drugs with a high risk of DILI; Category 0: drugs with no risk of DILI. The output value is the probability of being toxic.",
        "▪ Category 1: Ames positive(+); Category 0: Ames negative(-);<br/>▪ The output value is the probability of being toxic.",
        "▪ Category 0: low-toxicity; Category 1: high-toxicity;<br/>▪ The output value is the probability of being highly toxic.",
        "▪ Maximum Recommended Daily Dose<br/>▪ Category 1: FDAMDD (+); Category 0: FDAMDD (-)<br/>▪ The output value is the probability of being positive.",
        "▪ Category 1: Sensitizer; Category 0: Non-sensitizer;<br/>▪ The output value is the probability of being sensitizer.",
        "▪ Category 1: carcinogens; Category 0: non-carcinogens;<br/>▪ The output value is the probability of being toxic.",
        "▪ Category 1: corrosives ; Category 0: noncorrosives<br/>▪ The output value is the probability of being corrosives.",
        "▪ Category 1: irritants ; Category 0: nonirritants<br/>▪ The output value is the probability of being irritants.",
        "▪ Category 1: respiratory toxicants; Category 0: respiratory nontoxicants<br/>▪ The output value is the probability of being toxic.",
    ],
    style=styles["BodyText"],
)
toxicity_property = get_Para(
    [
        "hERG Blockers",
        "H-HT",
        "DILI",
        "AMES Toxicity",
        "Rat Oral Acute Toxicity",
        "FDAMDD",
        "Skin Sensitization",
        "Carcinogencity",
        "Eye Corrosion",
        "Eye Irritation",
        "Respiratory Toxicity",
    ],
    style=styles["BodyText"],
)


env_comment = get_Para(
    [
        "▪ Bioconcentration factors are used for considering secondary poisoning potential and assessing risks to human health via the food chain.<br/>▪ The unit is −log10[(mg/L)/(1000*MW)]",
        "▪ Tetrahymena pyriformis 50 percent growth inhibition concentration<br/>▪ The unit is −log10[(mg/L)/(1000*MW)]",
        "▪ 96-hour fathead minnow 50 percent lethal concentration<br/>▪ The unit is −log10[(mg/L)/(1000*MW)]",
        "▪ 48-hour daphnia magna 50 percent lethal concentration<br/>▪ The unit is −log10[(mg/L)/(1000*MW)]",
    ],
    style=styles["BodyText"],
)
env_property = get_Para(
    [
        "Bioconcentration Factors",
        "IGC<sub>50</sub>",
        "LC<sub>50</sub>FM",
        "LC<sub>50</sub>DM",
    ],
    style=styles["BodyText"],
)

pathway_comment = get_Para(
    [
        "▪ Androgen receptor<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
        "▪ Androgen receptor ligand-binding domain<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
        "▪ Aryl hydrocarbon receptor<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.<br/>",
        "▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
        "▪ Estrogen receptor<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
        "▪ Estrogen receptor ligand-binding domain<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
        "▪ Peroxisome proliferator-activated receptor gamma<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
        "▪ Antioxidant response element<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
        "▪ ATPase family AAA domain-containing protein 5<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
        "▪ Heat shock factor response element<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
        "▪ Mitochondrial membrane potential<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
        "▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
    ],
    style=styles["BodyText"],
)
pathway_property = get_Para(
    [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-<br/>gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ],
    style=styles["BodyText"],
)


def get_range(value):
    if 0 <= value <= 0.3:
        return '<font color="green">●</font>'
    elif 0.3 < value <= 0.7:
        return '<font color="orange">●</font>'
    else:
        return '<font color="red">●</font>'


def get_absor_decision(absorption_value):
    result = ["-"] * len(absorption_value)

    if absorption_value[0] <= -5.15:
        result[0] = Paragraph('<font color="red">●</font>', decision_style)
    else:
        result[0] = Paragraph('<font color="green">●</font>', decision_style)
    if absorption_value[1] <= 2e-6:
        result[1] = Paragraph('<font color="red">●</font>', decision_style)
    else:
        result[1] = Paragraph('<font color="green">●</font>', decision_style)
    for _, item in enumerate(absorption_value[2 : len(absorption_value)]):
        result[_ + 2] = Paragraph(get_range(item), decision_style)
    return result


def get_dis_decision(distribution_value):
    result = ["-"] * len(distribution_value)

    if float(distribution_value[0]) >= 90:
        result[0] = Paragraph('<font color="red">●</font>', decision_style)
    else:
        result[0] = Paragraph('<font color="green">●</font>', decision_style)
    if 0.04 < distribution_value[1] < 20:
        result[1] = Paragraph('<font color="green">●</font>', decision_style)
    else:
        result[1] = Paragraph('<font color="red">●</font>', decision_style)
    result[2] = Paragraph(get_range(distribution_value[2]), decision_style)
    if float(distribution_value[3]) < 5:
        result[3] = Paragraph('<font color="red">●</font>', decision_style)
    else:
        result[3] = Paragraph('<font color="green">●</font>', decision_style)
    return result


def myFirstPage(canvas, doc):
    canvas.saveState()
    canvas.drawImage(
        os.path.join(settings.SITE_ROOT, "static") + "/home/img/icon.png",
        x=inch,
        y=PAGE_HEIGHT - inch - 26,
        width=160,
        height=42,
        mask="auto",
    )
    # canvas.setFont('Lobster', 16)
    # canvas.drawCentredString(inch * 2 + 40, PAGE_HEIGHT - inch, doc.title)
    canvas.setFont("Times-Roman", 12)
    canvas.drawRightString(PAGE_WIDTH - inch, PAGE_HEIGHT - inch - 28, doc.smiles)
    canvas.setStrokeColor(Color(0, 0, 0, alpha=0.5))
    canvas.line(inch, PAGE_HEIGHT - 110, PAGE_WIDTH - inch, PAGE_HEIGHT - 110)
    canvas.setFont("Times-Roman", 9)
    canvas.drawString(inch, 0.75 * inch, "Page %d %s" % (doc.page, pageinfo))
    canvas.setFont("Times-Roman", 80)
    canvas.setFillAlpha(0.05)
    canvas.rotate(45)
    canvas.drawCentredString((PAGE_WIDTH / 2) * 1.5, 0, "OptADMET")
    # canvas.drawString(inch, 0.75 * inch, "First Page")
    canvas.restoreState()


def myLaterPages(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.drawString(inch, 0.75 * inch, "Page %d %s" % (doc.page, pageinfo))
    canvas.setFont("Times-Roman", 80)
    canvas.setFillAlpha(0.05)
    canvas.rotate(45)
    canvas.drawCentredString((PAGE_WIDTH / 2) * 1.5, 0, "OptADMET")
    canvas.restoreState()


def go(
    smiles,
    physicochemical,
    absorption,
    distribution,
    metabolism,
    excretion,
    toxicity,
    env,
    pathway,
    filepath,
):
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    doc.title = "OptADMET"
    doc.smiles = smiles
    story = []
    # Part 1
    physicochemical_table = Table(
        physicochemical,
        spaceBefore=2,
        colWidths=[doc.width * 0.2, doc.width * 0.2, doc.width * 0.6],
    )
    physicochemical_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )

    # Part 3
    absorption_table = Table(
        absorption,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.18,
            doc.width * 0.13,
            doc.width * 0.14,
            doc.width * 0.55,
        ],
    )
    absorption_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )

    # Part 4
    distribution_table = Table(
        distribution,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.15,
            doc.width * 0.15,
            doc.width * 0.15,
            doc.width * 0.55,
        ],
    )
    distribution_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )
    # Part 5
    metabolism_table = Table(
        metabolism,
        spaceBefore=2,
        colWidths=[doc.width * 0.2, doc.width * 0.2, doc.width * 0.6],
    )
    metabolism_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )

    # Part 6
    excretion_table = Table(
        excretion,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.15,
            doc.width * 0.15,
            doc.width * 0.15,
            doc.width * 0.55,
        ],
    )
    excretion_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )
    # Part 7
    toxicity_table = Table(
        toxicity,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.15,
            doc.width * 0.15,
            doc.width * 0.15,
            doc.width * 0.55,
        ],
    )
    toxicity_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )

    # Part 8
    env_table = Table(
        env,
        spaceBefore=2,
        colWidths=[doc.width * 0.2, doc.width * 0.15, doc.width * 0.65],
    )
    env_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )
    # Part 9
    pathway_table = Table(
        pathway,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.18,
            doc.width * 0.12,
            doc.width * 0.14,
            doc.width * 0.56,
        ],
    )
    pathway_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )
    story.append(Spacer(width=0, height=30))

    story.append(Paragraph("1. Physicochemical Property", style=styles["Heading2"]))
    story.append(physicochemical_table)

    story.append(Paragraph("2. Absorption", style=styles["Heading2"]))
    story.append(absorption_table)

    # story.append(Spacer(width=0, height=120))
    story.append(PageBreak())

    story.append(Paragraph("3. Distribution", style=styles["Heading2"]))
    story.append(distribution_table)

    story.append(Paragraph("4. Metabolism", style=styles["Heading2"]))
    story.append(metabolism_table)

    story.append(PageBreak())

    # story.append(Spacer(width=0, height=50))

    story.append(Paragraph("5. Excretion", style=styles["Heading2"]))
    story.append(excretion_table)

    story.append(Paragraph("6. Toxicity", style=styles["Heading2"]))
    story.append(toxicity_table)

    story.append(Paragraph("7. Environmental toxicity", style=styles["Heading2"]))
    story.append(env_table)

    story.append(Paragraph("8. Tox21 pathway", style=styles["Heading2"]))
    story.append(pathway_table)

    doc.build(story, onFirstPage=myFirstPage, onLaterPages=myLaterPages)


def get_excret_decision(excretion_value):
    result = ["-"] * len(excretion_value)
    if excretion_value[0] < 5:
        result[0] = Paragraph('<font color="red">●</font>', decision_style)
    else:
        result[0] = Paragraph('<font color="green">●</font>', decision_style)
    return result


def get_toxicity_decision(toxicity_value):
    result = ["-"] * len(toxicity_value)
    for _, item in enumerate(toxicity_value):
        result[_] = Paragraph(get_range(item), decision_style)
    return result


def get_pathway_decision(pathway_value):
    result = ["-"] * len(pathway_value)
    for _, item in enumerate(pathway_value):
        result[_] = Paragraph(get_range(item), decision_style)
    return result


def gen_pdf(row_data, filepath):
    # Part 1
    physicochemical_value = [
        row_data[item].round(3) for item in ["LogS", "LogP", "LogD"]
    ]
    physicochemical_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("Value", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    physicochemical = np.array(
        [physicochemical_property, physicochemical_value, physicochemical_comment]
    ).T
    physicochemical = np.vstack((physicochemical_header, physicochemical)).tolist()
    # Part 2
    absorption_value = [
        row_data[item]
        for item in ["Caco-2", "MDCK", "Pgp-inh", "Pgp-sub", "HIA", "F(20%)", "F(30%)"]
    ]
    round_absorption_value = [
        row_data[item].round(3)
        for item in ["Caco-2", "MDCK", "Pgp-inh", "Pgp-sub", "HIA", "F(20%)", "F(30%)"]
    ]
    absorption_decision = get_absor_decision(absorption_value)
    absorption_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("Value", header_center_style),
            get_Para("Decision", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    absorption = np.array(
        [
            absorption_property,
            round_absorption_value,
            absorption_decision,
            absorption_comment,
        ]
    ).T
    absorption = np.vstack((absorption_header, absorption)).tolist()
    # Part 3
    distribution_value = [
        row_data[item].round(3) for item in ["PPB", "VDss", "BBB", "Fu"]
    ]
    distribution_decision = get_dis_decision(distribution_value)
    distribution_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("Value", header_center_style),
            get_Para("Decision", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    distribution = np.array(
        [
            distribution_property,
            distribution_value,
            distribution_decision,
            distribution_comment,
        ]
    ).T
    distribution = np.vstack((distribution_header, distribution)).tolist()
    # Part 4
    metabolism_value = [
        row_data[item].round(3)
        for item in [
            "CYP1A2-inh",
            "CYP1A2-sub",
            "CYP2C19-inh",
            "CYP2C19-sub",
            "CYP2C9-inh",
            "CYP2C9-sub",
            "CYP2D6-inh",
            "CYP2D6-sub",
            "CYP3A4-inh",
            "CYP3A4-sub",
        ]
    ]
    metabolism_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("Value", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    metabolism = np.array([metabolism_property, metabolism_value, metabolism_comment]).T
    metabolism = np.vstack((metabolism_header, metabolism)).tolist()
    # Part 5
    excretion_value = [row_data[item].round(3) for item in ["CL", "T12"]]
    excretion_decision = get_excret_decision(excretion_value)
    excretion_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("Value", header_center_style),
            get_Para("Decision", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    excretion = np.array(
        [excretion_property, excretion_value, excretion_decision, excretion_comment]
    ).T
    excretion = np.vstack((excretion_header, excretion)).tolist()
    # Part 7
    toxicity_value = [
        row_data[item].round(3)
        for item in [
            "hERG",
            "H-HT",
            "DILI",
            "Ames",
            "ROA",
            "FDAMDD",
            "SkinSen",
            "Carcinogenicity",
            "EC",
            "EI",
            "Respiratory",
        ]
    ]
    toxicity_decision = get_toxicity_decision(toxicity_value)
    toxicity_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("Value", header_center_style),
            get_Para("Decision", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    toxicity = np.array(
        [toxicity_property, toxicity_value, toxicity_decision, toxicity_comment]
    ).T
    toxicity = np.vstack((toxicity_header, toxicity)).tolist()
    # Part 8
    env_value = [row_data[item].round(3) for item in ["BCF", "IGC50", "LC50", "LC50DM"]]
    env_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("Value", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    env = np.array([env_property, env_value, env_comment]).T
    env = np.vstack((env_header, env)).tolist()
    # Part 9
    pathway_value = [
        row_data[item].round(3)
        for item in [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ]
    ]
    pathway_decision = get_pathway_decision(pathway_value)
    pathway_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("Value", header_center_style),
            get_Para("Decision", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    pathway = np.array(
        [pathway_property, pathway_value, pathway_decision, pathway_comment]
    ).T
    pathway = np.vstack((pathway_header, pathway)).tolist()

    go(
        row_data["smiles"],
        physicochemical,
        absorption,
        distribution,
        metabolism,
        excretion,
        toxicity,
        env,
        pathway,
        filepath,
    )


def d_download(request):
    if request.method == "POST":
        filename = request.POST.get("csv_filename")
        index = request.POST.get("index")
        # filename = filepath.split('/')[-1]
        pdf_filepath = (
            os.path.join(settings.SITE_ROOT, "static/files/d_pdf/")
            + filename
            + "_"
            + str(index)
            + ".pdf"
        )
        if not os.path.exists(pdf_filepath):
            csv_filepath = (
                os.path.join(settings.SITE_ROOT, "static/files/result/tmp/")
                + filename
                + ".csv"
            )
            data = pd.read_csv(csv_filepath)
            gen_pdf2(data, pdf_filepath, filename, index)
        buffer = open(pdf_filepath, "rb")
        return FileResponse(
            buffer, as_attachment=True, filename=pdf_filepath.split("/")[-1]
        )
    else:
        return HttpResponseRedirect(reverse("home:index"))


def logd(ori, gen):
    if 1 <= gen <= 3:
        return True
    else:
        return False


def logp(ori, gen):
    if 0 <= gen <= 3:
        return True
    else:
        return False


def logs(ori, gen):
    if -4 <= gen <= 0.5:
        return True
    else:
        return False


def caco_2(ori, gen):
    if -5.15 <= gen:
        return True
    else:
        return False


def pgp_inhibitor(ori, gen):
    if 0 <= gen <= 0.3:
        return True
    else:
        return False


def pgp_substrate(ori, gen):
    if 0 <= gen <= 0.3:
        return True
    else:
        return False


def f20(ori, gen):
    if 0 <= gen <= 0.3:
        return True
    else:
        return False


def f30(ori, gen):
    if 0 <= gen <= 0.3:
        return True
    else:
        return False


def PPB(ori, gen):
    if gen < 90:
        return True
    else:
        return False


def BBB(ori, gen):
    if 0 <= gen <= 0.3:
        return True
    else:
        return False


def vdss(ori, gen):
    if 0.04 <= gen <= 20:
        return True
    else:
        return False


def inhibitor_1a2(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def inhibitor_2c19(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def inhibitor_2c9(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def inhibitor_2d6(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def substrate_2d6(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def inhibitor_3a4(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def t12(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def ames(ori, gen):
    if 0 <= gen <= 0.3:
        return True
    else:
        return False


def bcf(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def dili(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def ec(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def ei(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def fdamdd(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def hht(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def herg(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def igc50(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def nr_ahr(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def nr_ar(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def nr_ar_lbd(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def respiratory(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def sr_are(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


def sr_mmp(ori, gen):
    if gen <= 0.5:
        return True
    else:
        return False


property2func = {
    "logd": logd,
    "logp": logp,
    "logs": logs,
    "Caco-2": caco_2,
    "pgp-inh": pgp_inhibitor,
    "pgp-sub": pgp_substrate,
    "f20": f20,
    "f30": f30,
    "PPB": PPB,
    "BBB": BBB,
    "vdss": vdss,
    "cyp1a2-inh": inhibitor_1a2,
    "cyp2c19-inh": inhibitor_2c19,
    "cyp2c9-inh": inhibitor_2c9,
    "cyp2d6-inh": inhibitor_2d6,
    "cyp2d6-sub": substrate_2d6,
    "cyp3a4-inh": inhibitor_3a4,
    "t12": t12,
    "AMES": ames,
    "BCF": bcf,
    "Dili": dili,
    "ec": ec,
    "ei": ei,
    "fdamdd": fdamdd,
    "h-ht2": hht,
    "herg": herg,
    "igc50": igc50,
    "nr-ahr": nr_ahr,
    "nr-ar": nr_ar,
    "nr-ar-lbd": nr_ar_lbd,
    "respiratory": respiratory,
    "sr-are": sr_are,
    "sr-mmp": sr_mmp,
}


def final_screening(request):
    if request.method == "POST":
        file_path = request.POST.get("file_path")
        property = request.POST.get("property").split("_")
        select_property = request.POST.get("select").split(",")
        property = [idx2property[int(item)] for item in property]
        property = property + select_property if select_property != [""] else property
        property = list(set(property))
        properties = "_".join([str(property2idx[item]) for item in property])
        # print(properties)
        result_path = (
            os.path.join(settings.SITE_ROOT, "static")
            + "/files/result/tmp/"
            + file_path
            + ".csv"
        )
        datas = pd.read_csv(result_path)
        result_datas = pd.DataFrame(columns=datas.columns)
        result_datas = result_datas.append(datas.iloc[0], ignore_index=True)
        for _, row in datas.iterrows():
            if _ == 0:
                continue
            flags = [
                property2func[item](
                    row[property2header[item]], row[property2header[item]]
                )
                for item in property
            ]
            if False in flags:
                continue
            else:
                result_datas = result_datas.append(row, ignore_index=True)
        result_path = (
            os.path.join(settings.SITE_ROOT, "static")
            + "/files/final_result/tmp/"
            + file_path
            + ".csv"
        )
        result_datas.to_csv(result_path, index=False)
        return redirect(
            reverse(
                "home:final_result",
                kwargs={"filename": file_path, "properties": properties},
            )
        )
    else:
        return redirect(reverse("home:index"))


def bar(row, cols, mol, model):
    for col in cols:
        smart = row[col]
        smart = re.sub("\d+\*", "!#1", smart)
        patt = model[smart]

        if mol.HasSubstructMatch(patt):
            return True

    return False


def handle_fragment(fragment):
    # print(fragment, type(fragment))
    return re.sub("\d+\*", "!#1", fragment) if not isinstance(fragment, float) else ''


def handle_mol(mol):
    return Chem.AddHs(mol)


def handle_reaction(reaction):
    return ">>".join(reaction.split(">>")[::-1])


def get_patt(smarts):
    patt = Chem.MolFromSmarts(smarts)
    patt.UpdatePropertyCache(strict=False)
    patt_ = Chem.AddHs(patt, addCoords=True)
    return patt_


def _generatePkl(data, pattFile):
    model = {}
    cols = ["left_fragment", "right_fragment"]
    for col in cols:
        for smart in data[col].values:
            smart = re.sub("\d+\*", "!#1", smart)
            if smart not in model:
                patt = Chem.MolFromSmarts(smart)
                patt.UpdatePropertyCache(strict=False)
                patt_ = Chem.AddHs(patt, addCoords=True)
                model[smart] = patt_
    out = cPickle.dumps(model, protocol=-1)
    with gzip.open(pattFile, "wb") as f_out:
        f_out.write(out)
    f_out.close()
    return None


from scopy.ScoPretreat import pretreat


def convert_fragment_smart(left, right, pattMap):
    # s = pretreat.StandardizeMol()
    # # 加氢
    # left_fragment_smi_H = Chem.MolToSmiles(s.addhs(Chem.MolFromSmiles(left)))
    # left_fragment_H = Chem.MolFromSmiles(left_fragment_smi_H)
    # left_smarts = Chem.MolToSmarts(left_fragment_H)
    # # 标准化为反应smart
    # if "[2#0]" in left_smarts:
    #     left_smarts = left_smarts.replace("[2#0]", "[*:2]")
    # left_smarts = left_smarts.replace("[1#0]", "[*:1]")
    #  # 加氢
    # right_fragment_smi_H = Chem.MolToSmiles(s.addhs(Chem.MolFromSmiles(right)))
    # right_fragment_H = Chem.MolFromSmiles(right_fragment_smi_H)
    # # 标准化为反应smart
    # right_smart = Chem.MolToSmarts(right_fragment_H)
    # right_smart = right_smart.replace('[1#0]', '[*:1]')
    # if '[2#0]' in right_smart:
    #     right_smart = right_smart.replace('[2#0]', '[*:2]')
    left_smarts = pattMap[left]
    right_smarts = pattMap[right]

    return left_smarts + '>>' + right_smarts



def get_transform(
    mol,
    global_data,
    dataset,
    left="Left Fragment",
    right="Right Fragment",
    trans="Transformation Reaction SMARTS",
):
    tmp = handle_mol(mol)

    name = "_sort"
    base_path = os.path.join(settings.SITE_ROOT, "static")
    pattFile = base_path + "/media/genmol/" + f"{dataset}{name}.pkl.gz"

    # if not os.path.exists(pattFile):
        # _generatePkl(global_data, pattFile)

    pattMap = cPickle.load(gzip.open(pattFile, "rb"))

    # fragment_l = global_data[left].map(
    #     lambda fragment: pattMap[handle_fragment(fragment)]
    # )
    # fragment_r = global_data[right].map(
    #     lambda fragment: pattMap[handle_fragment(fragment)]
    # )

    # trans = global_data[left].map(
    #     lambda fragment: convert_fragment_smart(fragment)
    # )
    global_data[trans] = global_data.apply(lambda x: convert_fragment_smart(x[left], x[right], pattMap), axis=1)

    # bo_l = fragment_l.map(lambda patt: tmp.HasSubstructMatch(patt))
    # bo_r = fragment_r.map(lambda patt: tmp.HasSubstructMatch(patt))

    def foo(mol, reaction, reverse):
        # if met [1*][H] smart, the implicit hydrogen should be added to mol
        if reaction.split('>>')[0] == "[*:1]-[H]":
            mol = Chem.AddHs(mol)

        if reverse:
            reaction = handle_reaction(reaction)  # match right-fragment

        # rxn = rdChemReactions.ReactionFromSmarts(reaction)
        # reacts = (mol,)
        # products = rxn.RunReactants(reacts)

        # if products:
        #     products = [i for j in products for i in j]
        #     smis = [Chem.MolToSmiles(Chem.RemoveHs(product)) for product in products]
        #     smis = list(set(smis))
        #     return smis
        # 将SMARTS反应转换为反应对象
        rxn = AllChem.ReactionFromSmarts(reaction)

        ps = rxn.RunReactants((mol,))
        if ps:
            gen_smi_set = set()
            left_fragment = reaction.split('>>')[1]
            pattern = r'\[\d+\*\]'  # 匹配形如 [1*]、[2*]、[3*] 等的字符串
            left_str = re.sub(pattern, '', Chem.MolToSmiles(Chem.MolFromSmarts(left_fragment)))
            left_stand = ''.join(c.lower() for c in left_str if c.isalpha())
            if left_stand == 'h':
                left_stand = left_stand + left_stand
            for p in ps:
                smi = Chem.MolToSmiles(Chem.RemoveHs(p[0]))
                if ''.join(c.lower() for c in smi if c.isalpha()) == left_stand:
                    pass
                else:   
                    gen_smi_set.add(smi)
                # gen_smi_set.add(Chem.MolToSmiles(p[0]))
            return list(gen_smi_set)

        return None  # no result

    def bar(reverse=False):
        out = global_data.copy()
        out["newSmiles"] = global_data[trans].map(lambda reaction: foo(mol, reaction, reverse))
        out["flag"] = "right" if reverse else "left"
        return out

    out1 = bar()
    out2 = bar(reverse=True)

    out = pd.concat([out1, out2])
    out = out[out["newSmiles"].notna()]  # remove nan
    out = out.sort_index()

    return out


def HighlightAtoms(mol, highlightAtoms, figsize=[300, 300], kekulize=True):
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
        """ """
        svg_words = (
            svg_words.replace("stroke-width:2px", "stroke-width:1.5px")
            .replace("font-size:17px", "font-size:15px")
            .replace("stroke-linecap:butt", "stroke-linecap:square")
            .replace("fill:#FFFFFF", "fill:none")
            .replace("svg:", "")
        )
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


def final_result_datasource(request):
    if request.method == "POST":
        draw = int(request.POST.get("draw"))  # 記錄操作次數
        start = int(request.POST.get("start"))  # 起始位置
        length = int(request.POST.get("length"))  # 每頁長度
        filename = request.POST.get("filename")
        ori_property = request.POST.get("property").split("_")
        order_col = request.POST.get('order[0][column]')
        order_dir = request.POST.get('order[0][dir]')
        property = [idx2property[int(item)] for item in ori_property]
        result_path = (
            os.path.join(settings.SITE_ROOT, "static")
            + "/files/final_result/tmp/"
            + filename
            + ".csv"
        )
        mol_datas = pd.read_csv(result_path)
        results = mol_datas[start + 1 : start + length + 1]
        counts = len(mol_datas) - 1
        datas = []
        # if order_col and order_dir == 'asc':
        #     results = results.sort_values("sascore")
        # elif order_col and order_dir == 'desc':
        #     results = results.sort_values("sascore", ascending=False)
        if order_col:
            if order_dir == 'asc':
                results = results.sort_values(idx2header[int(ori_property[int(order_col) - 4])])
            else:
                results = results.sort_values(idx2header[int(ori_property[int(order_col) - 4])], ascending=False)
        for idx, row in results.iterrows():
            res = dict()
            mol = Chem.MolFromSmiles(row["smiles"])
            if row["flag"] == "left":
                atom_index = mol.GetSubstructMatch(
                    (Chem.MolFromSmarts(handle_fragment(row["right_fragment"])))
                )
                rxn = AllChem.ReactionFromSmarts(row["transformation"], useSmiles=True)
            else:
                atom_index = mol.GetSubstructMatch(
                    (Chem.MolFromSmarts(handle_fragment(row["left_fragment"])))
                )
                rxn = AllChem.ReactionFromSmarts(
                    handle_reaction(row["transformation"]), useSmiles=True
                )
            res["structure"] = HighlightAtoms(
                mol, highlightAtoms=atom_index, figsize=[200, 200]
            )
            res["smiles"] = row["smiles"]
            res["structure_id"] = row["structure_id"]
            res["transformation"] = row["structure_id"].split("_")[-1]
            res["property"] = property
            res["mol_index"] = row["mol_index"]
            res['sascore'] = row['sascore']
            res["svg"] = HighlightReaction(rxn, highlightAtoms=())
            for item in property:
                res[item] = row[property2header[item]]
            datas.append(res)
        response = dict()
        response["draw"] = draw
        response["recordsTotal"] = counts
        response["recordsFiltered"] = counts
        response["data"] = datas
        return HttpResponse(json.dumps(response), content_type="application/json")
    else:
        return render(request, "checker/index.html", locals())


def convert_left_fragment_smart(left_fragment):
    s = pretreat.StandardizeMol()
    # 加氢
    left_fragment_smi_H = Chem.MolToSmiles(s.addhs(Chem.MolFromSmiles(left_fragment)))
    left_fragment_H = Chem.MolFromSmiles(left_fragment_smi_H)
    left_smarts = Chem.MolToSmarts(left_fragment_H)
    # 标准化为反应smart
    if '[2#0]' in left_smarts:
        left_smarts = left_smarts.replace('[2#0]', '[*:2]')
    left_smarts = left_smarts.replace('[1#0]', '[*:1]')
    return left_smarts

def convert_right_fragment_smart(right_fragment):
    s = pretreat.StandardizeMol()
    # 加氢
    right_fragment_smi_H = Chem.MolToSmiles(s.addhs(Chem.MolFromSmiles(right_fragment)))
    right_fragment_H = Chem.MolFromSmiles(right_fragment_smi_H)
    # 标准化为反应smart
    right_smart = Chem.MolToSmarts(right_fragment_H)
    right_smart = right_smart.replace('[1#0]', '[*:1]')
    if '[2#0]' in right_smart:
        right_smart = right_smart.replace('[2#0]', '[*:2]')
    return right_smart


def result_datasource(request):
    if request.method == "POST":
        draw = int(request.POST.get("draw"))  # 記錄操作次數
        start = int(request.POST.get("start"))  # 起始位置
        length = int(request.POST.get("length"))  # 每頁長度
        filename = request.POST.get("filename")
        ori_property = request.POST.get("property").split("_")
        order_col = request.POST.get('order[0][column]')
        order_dir = request.POST.get('order[0][dir]')
        property = [idx2property[int(item)] for item in ori_property]
        result_path = (
            os.path.join(settings.SITE_ROOT, "static")
            + "/files/result/tmp/"
            + filename
            + ".csv"
        )
        mol_datas = pd.read_csv(result_path)
        print(mol_datas)
        results = mol_datas[start + 1 : start + length + 1]
        counts = len(mol_datas) - 1
        datas = []
        # 判断是否点击了排序事件
        if order_dir == 'asc':
            if len(property) == 1:
                results = results.sort_values(idx2header[int(ori_property[0])])
            elif len(property) == 2:
                if order_col == '4':
                    results = results.sort_values(idx2header[int(ori_property[0])])
                elif order_col == '5':
                    results = results.sort_values(idx2header[int(ori_property[1])])
        else:
            if len(property) == 1:
                results = results.sort_values(idx2header[int(ori_property[0])], ascending=False)
            elif len(property) == 2:
                if order_col == '4':
                    results = results.sort_values(idx2header[int(ori_property[0])], ascending=False)
                elif order_col == '5':
                    results = results.sort_values(idx2header[int(ori_property[1])], ascending=False)
        # if order_col and order_dir == 'asc':
        #     results = results.sort_values("sascore")
        # elif order_col and order_dir == 'desc':
        #     results = results.sort_values("sascore", ascending=False)
        for idx, row in results.iterrows():
            res = dict()
            mol = Chem.MolFromSmiles(row["smiles"])
            if row["flag"] == "left":
                atom_index = mol.GetSubstructMatch(
                    # (Chem.MolFromSmarts(handle_fragment(row["right_fragment"])))
                    (Chem.MolFromSmarts(convert_left_fragment_smart(row["right_fragment"])))
                )
                rxn = AllChem.ReactionFromSmarts(row["transformation"], useSmiles=True)
            else:
                atom_index = mol.GetSubstructMatch(
                    (Chem.MolFromSmarts(handle_fragment(row["left_fragment"])))
                )
                rxn = AllChem.ReactionFromSmarts(
                    handle_reaction(row["transformation"]), useSmiles=True
                )
            res["structure"] = HighlightAtoms(
                mol, highlightAtoms=atom_index, figsize=[150, 150]
            )
            res["smiles"] = row["smiles"]
            res["structure_id"] = row["structure_id"]
            res["transformation"] = row["structure_id"].split("_")[-1]
            res["property"] = property
            res["mol_index"] = row["mol_index"]
            res["svg"] = HighlightReaction(rxn, highlightAtoms=(), figsize=[300, 150])
            res['sascore'] = row['sascore']
            for item in property:
                res[item] = row[property2header[item]]
            datas.append(res)
        response = dict()
        response["draw"] = draw
        response["recordsTotal"] = counts
        response["recordsFiltered"] = counts
        response["data"] = datas
        return HttpResponse(json.dumps(response), content_type="application/json")
        # dataset = request.POST.get('dataset')
        # smiles = request.POST.get('smiles')
        # if dataset == '1':
        #     ori_mol = Chem.MolFromSmiles(smiles)
        #     mol = Chem.AddHs(ori_mol)
        #     if not property:
        #         # 全部
        #         app = OptSearch(mol)
        #         ans = app.search("all", mode="all", dataset="exp")
        #         counts = len(ans)
        #         results = ans[start: start + length]
        #         # out = get_transform(ori_mol, results)
        #         # length = out.newSmiles.map(lambda x: len(x))
        #         datas = []
        #         for idx, row in results.iterrows():
        #             res = dict()
        #             property, str_type, id = row['Structure_global_ID'].split('_')
        #             res['idx'] = id
        #             datas.append(res)
        #         response = dict()
        #         response['draw'] = draw
        #         response['recordsTotal'] = counts
        #         response['recordsFiltered'] = counts
        #         response['data'] = datas
        #         return HttpResponse(json.dumps(response), content_type='application/json')
        #     else:
        #         results_all = Experi_Sortlist.objects.values('transformation', 'left_fragment',
        #                                                      'right_fragment', 'transformation_reaction_SMARTS',
        #                                                      'structure_global_id').all()
        #         for item in property:
        #             property_structure = results_all.filter(
        #                 experi_property_structure__property_id=int(item))
        #             results_all = property_structure
        #         result_pd = pd.DataFrame(list(results_all), columns=['transformation', 'left_fragment',
        #                                                              'right_fragment', 'transformation_reaction_SMARTS',
        #                                                              'structure_global_id'])
        #         model = {}
        #         cols = ['left_fragment', 'right_fragment']
        #         for col in cols:
        #             for smart in result_pd[col].values:
        #                 smart = re.sub("\d+\*", "!#1", smart)
        #                 if smart not in model:
        #                     patt = Chem.MolFromSmarts(smart)
        #                     patt.UpdatePropertyCache(strict=False)
        #                     patt_ = Chem.AddHs(patt, addCoords=True)
        #                     model[smart] = patt_
        #         bo = result_pd.apply(lambda row: bar(row, cols, mol, model), axis=1)
        #         ans = result_pd[bo]
        #         counts = len(ans)
        #         if len(ans) == 0:
        #             results = pd.DataFrame()
        #         else:
        #             results = ans[start: start + length]
        #         datas = []
        #         for idx, row in results.iterrows():
        #             res = dict()
        #             property, str_type, id = row['structure_global_id'].split('_')
        #             res['idx'] = id
        #             datas.append(res)
        #         response = dict()
        #         response['draw'] = draw
        #         response['recordsTotal'] = counts
        #         response['recordsFiltered'] = counts
        #         response['data'] = datas
        #         return HttpResponse(json.dumps(response), content_type='application/json')
        # else:
        #     mol = Chem.MolFromSmiles(smiles)
        #     mol = Chem.AddHs(mol)
        #     if not property:
        #         # 全部
        #         app = OptSearch(mol)
        #         ans = app.search("all", mode="all", dataset="pre")
        #         counts = len(ans)
        #         results = ans[start: start + length]
        #         datas = []
        #         for idx, row in results.iterrows():
        #             res = dict()
        #             property, str_type, id = row['Structure_global_ID'].split('_')
        #             res['idx'] = id
        #
        #             datas.append(res)
        #         response = dict()
        #         response['draw'] = draw
        #         response['recordsTotal'] = counts
        #         response['recordsFiltered'] = counts
        #         response['data'] = datas
        #         return HttpResponse(json.dumps(response), content_type='application/json')
        #     else:
        #         results_all = Expand_Sortlist.objects.values('transformation', 'left_fragment',
        #                                                      'right_fragment', 'structure_global_id').all()
        #         for item in property:
        #             property_structure = results_all.filter(expand_property_structure__property_id=int(item))
        #             results_all = property_structure
        #         result_pd = pd.DataFrame(list(results_all), columns=['transformation', 'left_fragment',
        #                                                              'right_fragment', 'structure_global_id'])
        #         model = {}
        #         cols = ['left_fragment', 'right_fragment']
        #         for col in cols:
        #             for smart in result_pd[col].values:
        #                 smart = re.sub("\d+\*", "!#1", smart)
        #                 if smart not in model:
        #                     patt = Chem.MolFromSmarts(smart)
        #                     patt.UpdatePropertyCache(strict=False)
        #                     patt_ = Chem.AddHs(patt, addCoords=True)
        #                     model[smart] = patt_
        #         bo = result_pd.apply(lambda row: bar(row, cols, mol, model), axis=1)
        #         ans = result_pd[bo]
        #         counts = len(ans)
        #         if len(ans) == 0:
        #             results = pd.DataFrame()
        #         else:
        #             results = ans[start: start + length]
        #         datas = []
        #         for idx, row in results.iterrows():
        #             res = dict()
        #             property, str_type, id = row['structure_global_id'].split('_')
        #             res['idx'] = id
        #             datas.append(res)
        #         response = dict()
        #         response['draw'] = draw
        #         response['recordsTotal'] = counts
        #         response['recordsFiltered'] = counts
        #         response['data'] = datas
        #         return HttpResponse(json.dumps(response), content_type='application/json')

    else:
        return render(request, "checker/index.html", locals())


value_style = ParagraphStyle(
    name="value_style", fontName=_baseFontName, spaceBefore=6, alignment=TA_CENTER
)


def s_myFirstPage(canvas, doc):
    canvas.saveState()
    canvas.drawImage(
        os.path.join(settings.SITE_ROOT, "static") + "/home/img/icon.png",
        x=inch,
        y=PAGE_HEIGHT - inch - 26,
        width=160,
        height=42,
        mask="auto",
    )
    # canvas.setFont('Lobster', 16)
    # canvas.drawCentredString(inch * 2 + 40, PAGE_HEIGHT - inch, doc.title)
    canvas.setFont("Times-Roman", 20)
    canvas.drawRightString(PAGE_WIDTH - inch, PAGE_HEIGHT - inch - 28, doc.title)
    canvas.setStrokeColor(Color(0, 0, 0, alpha=0.5))
    canvas.line(inch, PAGE_HEIGHT - 110, PAGE_WIDTH - inch, PAGE_HEIGHT - 110)
    canvas.setFont("Times-Roman", 9)
    canvas.drawString(inch, 0.75 * inch, "Page %d %s" % (doc.page, pageinfo))
    canvas.setFont("Times-Roman", 80)
    canvas.setFillAlpha(0.05)
    canvas.rotate(45)
    canvas.drawCentredString((PAGE_WIDTH / 2) * 1.5, 0, "OptADMET")
    # canvas.drawString(inch, 0.75 * inch, "First Page")
    canvas.restoreState()


def logs_dec(value):
    if -4 <= value <= 0.5:
        return 0
    elif -5 < value < -4 or 0.5 < value < 1:
        return 1
    else:
        return 2


def logd_dec(value):
    if 0 <= value <= 3:
        return 0
    elif -1 < value < 0 or 3 < value < 4:
        return 1
    else:
        return 2


def logp_dec(value):
    if 1 <= value <= 3:
        return 0
    elif 0 < value < 1 or 3 < value < 4:
        return 1
    else:
        return 2


def Caco2_dec(value):
    if -5.15 <= value:
        return 0
    elif -7 < value < -5.15:
        return 1
    else:
        return 2


def MDCK_dec(value):
    if value >= 0.000002:
        return 0
    elif 0.000001 < value < 0.000002:
        return 1
    else:
        return 2


def pgpinh_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def pgpsub_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def HIA_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def f20_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def f30_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def PPB_dec(value):
    if value <= 90:
        return 0
    elif 90 < value < 95:
        return 1
    else:
        return 2


def BBB_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def vdss_dec(value):
    if 0.04 <= value <= 20:
        return 0
    elif 0.02 < value < 0.04 or 20 < value < 50:
        return 1
    else:
        return 2


def Fu_dec(value):
    if 5 <= value:
        return 0
    elif 4 < value < 5:
        return 1
    else:
        return 2


def cyp1a2inh_dec(value):
    if 0 <= value <= 0.4:
        return 0
    elif 0.4 < value <= 0.7:
        return 1
    else:
        return 2


def cyp1a2sub_dec(value):
    if 0 <= value <= 0.4:
        return 0
    elif 0.4 < value <= 0.7:
        return 1
    else:
        return 2


def cyp2c19inh_dec(value):
    if 0 <= value <= 0.4:
        return 0
    elif 0.4 < value <= 0.7:
        return 1
    else:
        return 2


def cyp2c19sub_dec(value):
    if 0 <= value <= 0.4:
        return 0
    elif 0.4 < value <= 0.7:
        return 1
    else:
        return 2


def cyp2c9inh_dec(value):
    if 0 <= value <= 0.4:
        return 0
    elif 0.4 < value <= 0.7:
        return 1
    else:
        return 2


def cyp2c9sub_dec(value):
    if 0 <= value <= 0.4:
        return 0
    elif 0.4 < value <= 0.7:
        return 1
    else:
        return 2


def cyp2d6inh_dec(value):
    if 0 <= value <= 0.4:
        return 0
    elif 0.4 < value <= 0.7:
        return 1
    else:
        return 2


def cyp2d6sub_dec(value):
    if 0 <= value <= 0.4:
        return 0
    elif 0.4 < value <= 0.7:
        return 1
    else:
        return 2


def cyp3a4inh_dec(value):
    if 0 <= value <= 0.4:
        return 0
    elif 0.4 < value <= 0.7:
        return 1
    else:
        return 2


def cyp3a4sub_dec(value):
    if 0 <= value <= 0.4:
        return 0
    elif 0.4 < value <= 0.7:
        return 1
    else:
        return 2


def CL_dec(value):
    if value >= 5:
        return 0
    elif 4 < value < 5:
        return 1
    else:
        return 2


def t12_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def AMES_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def ROA_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def Respiratory_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def Carcinogenicity_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def SkinSen_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def Dili_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def ec_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def ei_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def fdamdd_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def hht2_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def herg_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def nrahr_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def nrar_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def nrarlbd_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def nraromatase_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def nrer_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def nrerlbd_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def nrppargamma_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def srare_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def sratad5_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def srhse_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def srmmp_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


def srp53_dec(value):
    if 0 <= value <= 0.3:
        return 0
    elif 0.3 < value <= 0.7:
        return 1
    else:
        return 2


value2decision = {
    "logs": logs_dec,
    "logd": logd_dec,
    "logp": logp_dec,
    "Caco2": Caco2_dec,
    "MDCK": MDCK_dec,
    "pgpinh": pgpinh_dec,
    "pgpsub": pgpsub_dec,
    "HIA": HIA_dec,
    "f20": f20_dec,
    "f30": f30_dec,
    "PPB": PPB_dec,
    "BBB": BBB_dec,
    "vdss": vdss_dec,
    "Fu": Fu_dec,
    "cyp1a2inh": cyp1a2inh_dec,
    "cyp1a2sub": cyp1a2sub_dec,
    "cyp2c19inh": cyp2c19inh_dec,
    "cyp2c19sub": cyp2c19sub_dec,
    "cyp2c9inh": cyp2c9inh_dec,
    "cyp2c9sub": cyp2c9sub_dec,
    "cyp2d6inh": cyp2d6inh_dec,
    "cyp2d6sub": cyp2d6sub_dec,
    "cyp3a4inh": cyp3a4inh_dec,
    "cyp3a4sub": cyp3a4sub_dec,
    "CL": CL_dec,
    "t12": t12_dec,
    "AMES": AMES_dec,
    "ROA": ROA_dec,
    "Respiratory": Respiratory_dec,
    "Carcinogenicity": Carcinogenicity_dec,
    "SkinSen": SkinSen_dec,
    "Dili": Dili_dec,
    "ec": ec_dec,
    "ei": ei_dec,
    "fdamdd": fdamdd_dec,
    "hht2": hht2_dec,
    "herg": herg_dec,
    "nrahr": nrahr_dec,
    "nrar": nrar_dec,
    "nrarlbd": nrarlbd_dec,
    "nraromatase": nraromatase_dec,
    "nrer": nrer_dec,
    "nrerlbd": nrerlbd_dec,
    "nrppargamma": nrppargamma_dec,
    "srare": srare_dec,
    "sratad5": sratad5_dec,
    "srhse": srhse_dec,
    "srmmp": srmmp_dec,
    "srp53": srp53_dec,
}


header2dec = {
    "LogS": "logs",
    "LogD": "logd",
    "LogP": "logp",
    "Caco-2": "Caco2",
    "MDCK": "MDCK",
    "Pgp-inh": "pgpinh",
    "Pgp-sub": "pgpsub",
    "HIA": "HIA",
    "F(20%)": "f20",
    "F(30%)": "f30",
    "PPB": "PPB",
    "BBB": "BBB",
    "VDss": "vdss",
    "Fu": "Fu",
    "CYP1A2-inh": "cyp1a2inh",
    "CYP1A2-sub": "cyp1a2sub",
    "CYP2C19-inh": "cyp2c19inh",
    "CYP2C19-sub": "cyp2c19sub",
    "CYP2C9-inh": "cyp2c9inh",
    "CYP2C9-sub": "cyp2c9sub",
    "CYP2D6-inh": "cyp2d6inh",
    "CYP2D6-sub": "cyp2d6sub",
    "CYP3A4-inh": "cyp3a4inh",
    "CYP3A4-sub": "cyp3a4sub",
    "CL": "CL",
    "T12": "t12",
    "Ames": "AMES",
    "ROA": "ROA",
    "Respiratory": "Respiratory",
    "Carcinogenicity": "Carcinogenicity",
    "SkinSen": "SkinSen",
    "DILI": "Dili",
    "EC": "ec",
    "EI": "ei",
    "FDAMDD": "fdamdd",
    "H-HT": "hht2",
    "hERG": "herg",
    "NR-AhR": "nrahr",
    "NR-AR": "nrar",
    "NR-AR-LBD": "nrarlbd",
    "NR-Aromatase": "nraromatase",
    "NR-ER": "nrer",
    "NR-ER-LBD": "nrerlbd",
    "NR-PPAR-gamma": "nrppargamma",
    "SR-ARE": "srare",
    "SR-ATAD5": "sratad5",
    "SR-HSE": "srhse",
    "SR-MMP": "srmmp",
    "SR-p53": "srp53",
}

dec2header = {
    "logs": "LogS",
    "logd": "LogD",
    "logp": "LogP",
    "Caco2": "Caco-2",
    "MDCK": "MDCK",
    "pgpinh": "Pgp-inh",
    "pgpsub": "Pgp-sub",
    "HIA": "HIA",
    "f20": "F(20%)",
    "f30": "F(30%)",
    "PPB": "PPB",
    "BBB": "BBB",
    "vdss": "VDss",
    "Fu": "Fu",
    "cyp1a2inh": "CYP1A2-inh",
    "cyp1a2sub": "CYP1A2-sub",
    "cyp2c19inh": "CYP2C19-inh",
    "cyp2c19sub": "CYP2C19-sub",
    "cyp2c9inh": "CYP2C9-inh",
    "cyp2c9sub": "CYP2C9-sub",
    "cyp2d6inh": "CYP2D6-inh",
    "cyp2d6sub": "CYP2D6-sub",
    "cyp3a4inh": "CYP3A4-inh",
    "cyp3a4sub": "CYP3A4-sub",
    "CL": "CL",
    "t12": "T12",
    "AMES": "Ames",
    "ROA": "ROA",
    "Respiratory": "Respiratory",
    "Carcinogenicity": "Carcinogenicity",
    "SkinSen": "SkinSen",
    "Dili": "DILI",
    "ec": "EC",
    "ei": "EI",
    "fdamdd": "FDAMDD",
    "hht2": "H-HT",
    "herg": "hERG",
    "nrahr": "NR-AhR",
    "nrar": "NR-AR",
    "nrarlbd": "NR-AR-LBD",
    "nraromatase": "NR-Aromatase",
    "nrer": "NR-ER",
    "nrerlbd": "NR-ER-LBD",
    "nrppargamma": "NR-PPAR-gamma",
    "srare": "SR-ARE",
    "sratad5": "SR-ATAD5",
    "srhse": "SR-HSE",
    "srmmp": "SR-MMP",
    "srp53": "SR-p53",
}


def gen_molimg(smiles, path):
    try:
        mol = Chem.MolFromSmiles(smiles)
        d = rdMolDraw2D.MolDraw2DCairo(300, 300)
        # d.drawOptions().useBWAtomPalette()
        # d.drawOptions().addStereoAnnotation = True
        d.DrawMolecule(mol)
        d.FinishDrawing()
        d.WriteDrawingText(path)
    except:
        pass


def gen_transimg(smarts, path):
    try:
        rxn = AllChem.ReactionFromSmarts(smarts, useSmiles=True)
        drawer = rdMolDraw2D.MolDraw2DCairo(360, 180)
        drawer.DrawReaction(rxn, highlightByReactant=True)
        # drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        with open(path, "wb+").write(svg) as f:
            pass
        # drawer.WriteDrawingText(path)
        # d.WriteDrawingText(path)
    except:
        pass


success_inter = {
    "LogS": "[-4,0.5]",
    "LogD": "[0,3]",
    "LogP": "[1,3]",
    "Caco-2": "[-5.15,+∞)",
    "MDCK": "[2e-6,+∞)",
    "Pgp-inh": "[0, 0.3]",
    "Pgp-sub": "[0, 0.3]",
    "HIA": "[0, 0.3]",
    "F(20%)": "[0, 0.3]",
    "F(30%)": "[0, 0.3]",
    "PPB": "[0,0.9]",
    "BBB": "[0, 0.3]",
    "VDss": "[0.04,20]",
    "Fu": "[5,+∞)",
    "CYP1A2-inh": "[0, 0.4]",
    "CYP1A2-sub": "[0, 0.4]",
    "CYP2C19-inh": "[0, 0.4]",
    "CYP2C19-sub": "[0, 0.4]",
    "CYP2C9-inh": "[0, 0.4]",
    "CYP2C9-sub": "[0, 0.4]",
    "CYP2D6-inh": "[0, 0.4]",
    "CYP2D6-sub": "[0, 0.4]",
    "CYP3A4-inh": "[0, 0.4]",
    "CYP3A4-sub": "[0, 0.4]",
    "CL": "[5,+∞)",
    "T12": "[0, 0.3]",
    "Ames": "[0, 0.3]",
    "ROA": "[0, 0.3]",
    "Respiratory": "[0, 0.3]",
    "Carcinogenicity": "[0, 0.3]",
    "SkinSen": "[0, 0.3]",
    "DILI": "[0, 0.3]",
    "EC": "[0, 0.3]",
    "EI": "[0, 0.3]",
    "FDAMDD": "[0, 0.3]",
    "H-HT": "[0, 0.3]",
    "hERG": "[0, 0.3]",
    "NR-AhR": "[0, 0.3]",
    "NR-AR": "[0, 0.3]",
    "NR-AR-LBD": "[0, 0.3]",
    "NR-Aromatase": "[0, 0.3]",
    "NR-ER": "[0, 0.3]",
    "NR-ER-LBD": "[0, 0.3]",
    "NR-PPAR-gamma": "[0, 0.3]",
    "SR-ARE": "[0, 0.3]",
    "SR-ATAD5": "[0, 0.3]",
    "SR-HSE": "[0, 0.3]",
    "SR-MMP": "[0, 0.3]",
    "SR-p53": "[0, 0.3]",
}

warning_inter = {
    "LogS": "(-5,-4)or(0.5,1)",
    "LogD": "(-1,0)or(3,4)",
    "LogP": "(0,1)or(3,4)",
    "Caco-2": "(-7,-5.15)",
    "MDCK": "(1e-6,2e-6)",
    "Pgp-inh": "(0.3,0.7]",
    "Pgp-sub": "(0.3,0.7]",
    "HIA": "(0.3,0.7]",
    "F(20%)": "(0.3,0.7]",
    "F(30%)": "(0.3,0.7]",
    "PPB": "(0.9,0.95)",
    "BBB": "(0.3,0.7]",
    "VDss": "(0.02,0.04)or(20,25)",
    "Fu": "(4,5)",
    "CYP1A2-inh": "(0.3,0.7]",
    "CYP1A2-sub": "(0.3,0.7]",
    "CYP2C19-inh": "(0.3,0.7]",
    "CYP2C19-sub": "(0.3,0.7]",
    "CYP2C9-inh": "(0.3,0.7]",
    "CYP2C9-sub": "(0.3,0.7]",
    "CYP2D6-inh": "(0.3,0.7]",
    "CYP2D6-sub": "(0.3,0.7]",
    "CYP3A4-inh": "(0.3,0.7]",
    "CYP3A4-sub": "(0.3,0.7]",
    "CL": "(4,5)",
    "T12": "(0.3,0.7]",
    "Ames": "(0.3,0.7]",
    "ROA": "(0.3,0.7]",
    "Respiratory": "(0.3,0.7]",
    "Carcinogenicity": "(0.3,0.7]",
    "SkinSen": "(0.3,0.7]",
    "DILI": "(0.3,0.7]",
    "EC": "(0.3,0.7]",
    "EI": "(0.3,0.7]",
    "FDAMDD": "(0.3,0.7]",
    "H-HT": "(0.3,0.7]",
    "hERG": "(0.3,0.7]",
    "NR-AhR": "(0.3,0.7]",
    "NR-AR": "(0.3,0.7]",
    "NR-AR-LBD": "(0.3,0.7]",
    "NR-Aromatase": "(0.3,0.7]",
    "NR-ER": "(0.3,0.7]",
    "NR-ER-LBD": "(0.3,0.7]",
    "NR-PPAR-gamma": "(0.3,0.7]",
    "SR-ARE": "(0.3,0.7]",
    "SR-ATAD5": "(0.3,0.7]",
    "SR-HSE": "(0.3,0.7]",
    "SR-MMP": "(0.3,0.7]",
    "SR-p53": "(0.3,0.7]",
}

danger_inter = {
    "LogS": "(-∞,-5)or(1,+∞)",
    "LogD": "(-∞,-1)or(4,+∞)",
    "LogP": "(-∞,0)or(4,+∞)",
    "Caco-2": "(-∞,-7)",
    "MDCK": "(-∞,1e-6)",
    "Pgp-inh": "(0.7,1]",
    "Pgp-sub": "(0.7,1]",
    "HIA": "(0.7,1]",
    "F(20%)": "(0.7,1]",
    "F(30%)": "(0.7,1]",
    "PPB": "(0.95,1]",
    "BBB": "(0.7,1]",
    "VDss": "(-∞,0.02)or(25,+∞)",
    "Fu": "(-∞,4)",
    "CYP1A2-inh": "(0.7,1]",
    "CYP1A2-sub": "(0.7,1]",
    "CYP2C19-inh": "(0.7,1]",
    "CYP2C19-sub": "(0.7,1]",
    "CYP2C9-inh": "(0.7,1]",
    "CYP2C9-sub": "(0.7,1]",
    "CYP2D6-inh": "(0.7,1]",
    "CYP2D6-sub": "(0.7,1]",
    "CYP3A4-inh": "(0.7,1]",
    "CYP3A4-sub": "(0.7,1]",
    "CL": "(-∞,4)",
    "T12": "(0.7,1]",
    "Ames": "(0.7,1]",
    "ROA": "(0.7,1]",
    "Respiratory": "(0.7,1]",
    "Carcinogenicity": "(0.7,1]",
    "SkinSen": "(0.7,1]",
    "DILI": "(0.7,1]",
    "EC": "(0.7,1]",
    "EI": "(0.7,1]",
    "FDAMDD": "(0.7,1]",
    "H-HT": "(0.7,1]",
    "hERG": "(0.7,1]",
    "NR-AhR": "(0.7,1]",
    "NR-AR": "(0.7,1]",
    "NR-AR-LBD": "(0.7,1]",
    "NR-Aromatase": "(0.7,1]",
    "NR-ER": "(0.7,1]",
    "NR-ER-LBD": "(0.7,1]",
    "NR-PPAR-gamma": "(0.7,1]",
    "SR-ARE": "(0.7,1]",
    "SR-ATAD5": "(0.7,1]",
    "SR-HSE": "(0.7,1]",
    "SR-MMP": "(0.7,1]",
    "SR-p53": "(0.7,1]",
}


def gen_s_pdf(filename):
    with open(
        os.path.join(settings.SITE_ROOT, "static/files/meta/") + filename + ".json"
    ) as f:
        meta_datas = json.load(f)
    json_path = (
        os.path.join(settings.SITE_ROOT, "static/files/summary_info/")
        + filename
        + ".json"
    )
    with open(json_path) as f:
        all_datas = json.load(f)
    datas = all_datas["generated"]
    ori_datas = all_datas["origin"]
    img_path = (
        os.path.join(settings.SITE_ROOT, "static/files/orimol_img/") + filename + ".png"
    )
    if not os.path.exists(img_path):
        gen_molimg(smiles=meta_datas["smiles"], path=img_path)
    # 生成原始分子图像
    ori_smiles = meta_datas["smiles"]

    pro2header = {
        "logd": "LogD",
        "logp": "LogP",
        "logs": "LogS",
        "Caco2": "Caco-2",
        "pgpinh": "Pgp-inh",
        "MDCK": "MDCK",
        "HIA": "HIA",
        "Fu": "Fu",
        "cyp1a2sub": "CYP1A2<br/>-sub",
        "cyp2c19sub": "CYP2C19<br/>-sub",
        "cyp2c9sub": "CYP2C9<br/>-sub",
        "cyp3a4sub": "CYP3A4<br/>-sub",
        "ROA": "Rat Oral Acute Toxicity",
        "Respiratory": "Respiratory Toxicity",
        "Carcinogenicity": "Carcinogencity",
        "SkinSen": "Skin Sensitization",
        "nraromatase": "NR-Aromatase",
        "nrer": "NR-ER",
        "nrerlbd": "NR-ER-LBD",
        "nrppargamma": "NR-PPAR-<br/>gamma",
        "sratad5": "SR-ATAD5",
        "srhse": "SR-HSE",
        "srp53": "SR-p53",
        "CL": "CL",
        "pgpsub": "Pgp-sub",
        "f20": "F<sub>20%</sub>",
        "f30": "F<sub>30%</sub>",
        "PPB": "PPB",
        "BBB": "BBB <br/>Penetration",
        "vdss": "VDss",
        "cyp1a2inh": "CYP1A2<br/>-inh",
        "cyp2c19inh": "CYP2C19<br/>-inh",
        "cyp2c9inh": "CYP2C9<br/>-inh",
        "cyp2d6inh": "CYP2D6<br/>-inh",
        "cyp2d6sub": "CYP2D6<br/>-sub",
        "cyp3a4inh": "CYP3A4<br/>-inh",
        "t12": "T<sub>1/2</sub>",
        "AMES": "Ames",
        "Dili": "DILI",
        "ec": "EC",
        "ei": "EI",
        "fdamdd": "FDAMDD",
        "hht2": "H-HT",
        "herg": "hERG",
        "nrahr": "NR-AhR",
        "nrar": "NR-AR",
        "nrarlbd": "NR-AR-LBD",
        "srare": "SR-ARE",
        "srmmp": "SR-MMP",
    }
    success_values = get_Para([datas[item]["0"] for item in datas.keys()], value_style)
    warning_values = get_Para([datas[item]["1"] for item in datas.keys()], value_style)
    failure_values = get_Para([datas[item]["2"] for item in datas.keys()], value_style)
    property = [pro2header[item] for item in datas.keys()]
    property = get_Para(property, style=styles["BodyText"])
    success_decision = [
        Paragraph('<font color="green">●</font>', decision_style)
        for item in range(len(datas))
    ]
    warning_decision = [
        Paragraph('<font color="yellow">●</font>', decision_style)
        for item in range(len(datas))
    ]
    failure_decision = [
        Paragraph('<font color="red">●</font>', decision_style)
        for item in range(len(datas))
    ]
    comment = get_Para(
        [
            "Log of the aqueous solubility. Optimal: -4~0.5 log mol/L",
            "Log of the octanol/water partition coefficient. Optimal: 0~3",
            "logP at physiological pH 7.4. Optimal: 1~3",
            "Optimal: higher than -5.15 Log unit",
            "▪ low permeability: < 2 × 10<super>−6</super> cm/s<br/>▪ medium permeability: 2–20 × 10<super>−6</super> cm/s<br/>▪ high passive permeability: > 20 × 10<super>−6</super> cm/s",
            "▪ Category 1: Inhibitor; Category 0: Non-inhibitor;<br/>▪ The output value is the probability of being Pgp-inhibitor",
            "▪ Category 1: substrate; Category 0: Non-substrate;<br/>▪ The output value is the probability of being Pgp-substrate",
            "▪ Human Intestinal Absorption<br/>▪ Category 1: HIA+( HIA < 30%); Category 0: HIA-( HIA < 30%); The output value is the probability of being HIA+",
            "▪ 20% Bioavailability<br/>▪ Category 1: F<sub>20%</sub>+ (bioavailability < 20%); Category 0: F<sub>20%</sub>- (bioavailability ≥ 20%); The output value is the probability of being F<sub>20%</sub>+",
            "▪ 30% Bioavailability<br/>▪ Category 1: F<sub>30%</sub>+ (bioavailability < 30%); Category 0: F<sub>30%</sub>- (bioavailability ≥ 30%); The output value is the probability of being F<sub>30%</sub>+",
            "▪ Plasma Protein Binding<br/>▪ Optimal: < 90%. Drugs with high protein-bound may have a low therapeutic index.",
            "▪ Blood-Brain Barrier Penetration<br/>▪ Category 1: BBB+; Category 0: BBB-; The output value is the probability of being BBB+",
            "▪ Volume Distribution<br/>▪ Optimal: 0.04-20L/kg",
            "▪ The fraction unbound in plasms<br/>▪ Low: <5%; Middle: 5~20%; High: > 20%",
            "▪ Category 1: Inhibitor; Category 0: Non-inhibitor;<br/>▪ The output value is the probability of being inhibitor.",
            "▪ Category 1: Substrate; Category 0: Non-substrate;<br/>▪ The output value is the probability of being substrate.",
            "▪ Category 1: Inhibitor; Category 0: Non-inhibitor;<br/>▪ The output value is the probability of being inhibitor.",
            "▪ Category 1: Substrate; Category 0: Non-substrate;<br/>▪ The output value is the probability of being substrate.",
            "▪ Category 1: Inhibitor; Category 0: Non-inhibitor;<br/>▪ The output value is the probability of being inhibitor.",
            "▪ Category 1: Substrate; Category 0: Non-substrate;<br/>▪ The output value is the probability of being substrate.",
            "▪ Category 1: Inhibitor; Category 0: Non-inhibitor;<br/>▪ The output value is the probability of being inhibitor.",
            "▪ Category 1: Substrate; Category 0: Non-substrate;<br/>▪ The output value is the probability of being substrate.",
            "▪ Category 1: Inhibitor; Category 0: Non-inhibitor;<br/>▪ The output value is the probability of being inhibitor.",
            "▪ Category 1: Substrate; Category 0: Non-substrate;<br/>▪ The output value is the probability of being substrate.",
            "▪ Clearance<br/>▪ High: >15 mL/min/kg; moderate: 5-15 mL/min/kg; low: <5 mL/min/kg",
            "▪ Category 1: long half-life ; Category 0: short half-life;<br/>▪ long half-life: >3h; short half-life: <3h<br/>▪ The output value is the probability of having long half-life.",
            "▪ Category 1: Ames positive(+); Category 0: Ames negative(-);<br/>▪ The output value is the probability of being toxic.",
            "▪ Category 0: low-toxicity; Category 1: high-toxicity;<br/>▪ The output value is the probability of being highly toxic.",
            "▪ Category 1: respiratory toxicants; Category 0: respiratory nontoxicants<br/>▪ The output value is the probability of being toxic.",
            "▪ Category 1: carcinogens; Category 0: non-carcinogens;<br/>▪ The output value is the probability of being toxic.",
            "▪ Category 1: Sensitizer; Category 0: Non-sensitizer;<br/>▪ The output value is the probability of being sensitizer.",
            "▪ Drug Induced Liver Injury.<br/>▪ Category 1: drugs with a high risk of DILI; Category 0: drugs with no risk of DILI. The output value is the probability of being toxic.",
            "▪ Category 1: corrosives ; Category 0: noncorrosives<br/>▪ The output value is the probability of being corrosives.",
            "▪ Category 1: irritants ; Category 0: nonirritants<br/>▪ The output value is the probability of being irritants.",
            "▪ Maximum Recommended Daily Dose<br/>▪ Category 1: FDAMDD (+); Category 0: FDAMDD (-)<br/>▪ The output value is the probability of being positive.",
            "▪ Human Hepatotoxicity<br/>▪ Category 1: H-HT positive(+); Category 0: H-HT negative(-);<br/>▪ The output value is the probability of being toxic.",
            "▪ Category 1: active; Category 0: inactive;<br/>▪ The output value is the probability of being active.",
            "▪ Aryl hydrocarbon receptor<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.<br/>",
            "▪ Androgen receptor<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
            "▪ Androgen receptor ligand-binding domain<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
            "▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
            "▪ Estrogen receptor<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
            "▪ Estrogen receptor ligand-binding domain<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
            "▪ Peroxisome proliferator-activated receptor gamma<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
            "▪ Antioxidant response element<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
            "▪ ATPase family AAA domain-containing protein 5<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
            "▪ Heat shock factor response element<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
            "▪ Mitochondrial membrane potential<br/>▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
            "▪ Category 1: actives ; Category 0: inactives;<br/>▪ The output value is the probability of being active.",
        ],
        style=styles["BodyText"],
    )
    header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("S.", header_center_style),
            get_Para("N.", header_center_style),
            get_Para("W.", header_center_style),
            get_Para("N.", header_center_style),
            get_Para("D.", header_center_style),
            get_Para("N.", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    table = np.array(
        [
            property,
            success_decision,
            success_values,
            warning_decision,
            warning_values,
            failure_decision,
            failure_values,
            comment,
        ]
    ).T
    table = np.vstack((header, table)).tolist()
    doc = SimpleDocTemplate(
        os.path.join(settings.SITE_ROOT, "static/files/s_pdf/") + filename + ".pdf",
        pagesize=A4,
    )
    doc.title = "Summary Information"
    story = []
    table = Table(
        table,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.14,
            doc.width * 0.06,
            doc.width * 0.08,
            doc.width * 0.06,
            doc.width * 0.08,
            doc.width * 0.06,
            doc.width * 0.08,
            doc.width * 0.44,
        ],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )

    class ImageText(Flowable):
        def __init__(self, img_path="", width=120, height=110):
            super().__init__()
            from reportlab.lib.units import inch

            self.height = height
            self.img_width = width
            self.img_path = img_path

        def draw(self):
            canvas = self.canv
            canvas.translate(0, 0)
            canvas.setFont("Times-Roman", 10)
            canvas.drawImage(
                self.img_path,
                x=0,
                y=-10,
                width=self.img_width,
                height=self.img_width,
                mask="auto",
            )
            if len(meta_datas["ori_idx"]) == 2:
                key1, key2 = meta_datas["ori_idx"].keys()
            else:
                key1 = list(meta_datas["ori_idx"].keys())[0]
            if len(meta_datas["ori_idx"]) == 2:
                canvas.drawString(
                    self.img_width + 15,
                    80,
                    str(key1) + ": " + str(success_inter[key1]),
                    charSpace="1",
                )

                canvas.drawString(
                    self.img_width + 110,
                    80,
                    str(key2) + ": " + str(success_inter[key2]),
                    charSpace="1",
                )
            else:
                canvas.drawString(
                    self.img_width + 15,
                    80,
                    str(key1) + ": " + str(success_inter[key1]),
                    charSpace="1",
                )

            canvas.setFillColor(Color(36 / 255, 118 / 255, 26 / 255))
            canvas.setStrokeColor(Color(36 / 255, 118 / 255, 26 / 255))
            canvas.circle(PAGE_WIDTH - inch - 190, 63, 3, fill=1)
            canvas.drawString(
                self.img_width + 15, 60, "S: " + str(success_inter[key1]), charSpace="1"
            )
            if len(meta_datas["ori_idx"]) == 2:
                canvas.drawString(
                    self.img_width + 110,
                    60,
                    "S: " + str(success_inter[key2]),
                    charSpace="1",
                )

            canvas.drawString(
                PAGE_WIDTH - inch - 180, 60, "Recommended Interval", charSpace=0.5
            )

            canvas.setFillColor(Color(241 / 255, 233 / 255, 54 / 255))
            canvas.setStrokeColor(Color(241 / 255, 233 / 255, 54 / 255))
            canvas.circle(PAGE_WIDTH - inch - 190, 43, 3, fill=1)
            canvas.drawString(
                self.img_width + 14, 40, "W: " + str(warning_inter[key1]), charSpace="1"
            )
            if len(meta_datas["ori_idx"]) == 2:
                canvas.drawString(
                    self.img_width + 109,
                    40,
                    "W: " + str(warning_inter[key2]),
                    charSpace="1",
                )

            canvas.drawString(
                PAGE_WIDTH - inch - 180, 40, "Warnning Interval", charSpace=0.5
            )

            canvas.setFillColor(Color(215 / 255, 18 / 255, 36 / 255))
            canvas.setStrokeColor(Color(215 / 255, 18 / 255, 36 / 255))
            canvas.circle(PAGE_WIDTH - inch - 190, 23, 3, fill=1)
            canvas.drawString(
                self.img_width + 15, 20, "D: " + str(danger_inter[key1]), charSpace="1"
            )
            if len(meta_datas["ori_idx"]) == 2:
                canvas.drawString(
                    self.img_width + 110,
                    20,
                    "D: " + str(danger_inter[key2]),
                    charSpace="1",
                )

            canvas.drawString(
                PAGE_WIDTH - inch - 180, 20, "Danger Interval", charSpace=0.5
            )

    story.append(Spacer(width=0, height=30))
    story.append(Paragraph("1. Original molecular profile", style=styles["Heading2"]))
    story.append(Paragraph(ori_smiles, style=styles["BodyText"]))
    story.append(ImageText(img_path))
    import math

    table_list = []
    dec2header = {
        "logs": "LogS",
        "logd": "LogD",
        "logp": "LogP",
        "Caco2": "Caco-2",
        "MDCK": "MDCK",
        "pgpinh": "Pgp-inh",
        "pgpsub": "Pgp-sub",
        "HIA": "HIA",
        "f20": "F(20%)",
        "f30": "F(30%)",
        "PPB": "PPB",
        "BBB": "BBB",
        "vdss": "VDss",
        "Fu": "Fu",
        "cyp1a2inh": "CYP1A2-inh",
        "cyp1a2sub": "CYP1A2-sub",
        "cyp2c19inh": "CYP2C19-inh",
        "cyp2c19sub": "CYP2C19-sub",
        "cyp2c9inh": "CYP2C9-inh",
        "cyp2c9sub": "CYP2C9-sub",
        "cyp2d6inh": "CYP2D6-inh",
        "cyp2d6sub": "CYP2D6-sub",
        "cyp3a4inh": "CYP3A4-inh",
        "cyp3a4sub": "CYP3A4-sub",
        "CL": "CL",
        "t12": "T12",
        "AMES": "Ames",
        "ROA": "ROA",
        "Respiratory": "Respiratory",
        "Carcinogenicity": "Carcinogenicity",
        "SkinSen": "SkinSen",
        "Dili": "DILI",
        "ec": "EC",
        "ei": "EI",
        "fdamdd": "FDAMDD",
        "hht2": "H-HT",
        "herg": "hERG",
        "nrahr": "NR-AhR",
        "nrar": "NR-AR",
        "nrarlbd": "NR-AR-LBD",
        "nraromatase": "NR-Aromatase",
        "nrer": "NR-ER",
        "nrerlbd": "NR-ER-LBD",
        "nrppargamma": "NR-PPAR-gamma",
        "srare": "SR-ARE",
        "sratad5": "SR-ATAD5",
        "srhse": "SR-HSE",
        "srmmp": "SR-MMP",
        "srp53": "SR-p53",
    }
    for item, value in ori_datas.items():
        table_list.append(Paragraph(dec2header[item], style=styles["BodyText"]))
        if value == 0:
            table_list.append(Paragraph('<font color="green">●</font>', decision_style))
        elif value == 1:
            table_list.append(
                Paragraph('<font color="yellow">●</font>', decision_style)
            )
        else:
            table_list.append(Paragraph('<font color="red">●</font>', decision_style))
    row_count = 8
    latest_number = (math.ceil(len(table_list) / row_count)) * row_count
    table_list[len(table_list) : latest_number] = ""
    for item in range(len(table_list), latest_number):
        table_list.append(Paragraph("", decision_style))
    table_list = np.array(table_list).reshape(-1, row_count).tolist()
    ori_table = Table(
        table_list,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.19,
            doc.width * 0.06,
            doc.width * 0.19,
            doc.width * 0.06,
            doc.width * 0.19,
            doc.width * 0.06,
            doc.width * 0.19,
            doc.width * 0.06,
        ],
    )
    ori_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )
    story.append(ori_table)

    bar_data = meta_datas["data"]
    for _, (key, value) in enumerate(bar_data.items()):
        if _ == 0:
            drawing = Drawing((PAGE_WIDTH - inch * 2) / 2, 220)
        else:
            drawing = Drawing((PAGE_WIDTH - inch * 2) / 2, 0)
        data = [tuple(value.values())]
        bc = VerticalBarChart()
        bc.x = 0
        if _ == 1:
            bc.x = (PAGE_WIDTH - inch * 2) / 2 + 20
        bc.y = 30
        bc.height = 160
        bc.width = (PAGE_WIDTH - inch * 2) / 2 - 20
        if len(bar_data) == 1:
            bc.width = PAGE_WIDTH - inch * 2
        bc.data = data
        bc.strokeColor = None
        bc.bars[0].fillColor = colors.skyblue
        if _ == 1:
            bc.bars[0].fillColor = colors.indianred
        # bc.valueAxis.valueMin = 0
        # bc.valueAxis.valueMax = 50
        # bc.valueAxis.valueStep = 10
        # bc.title = key
        bc.barLabelFormat = "%s"
        bc.barLabels.dy = 10
        bc.categoryAxis.labels.boxAnchor = "ne"
        bc.categoryAxis.labels.dx = 8
        bc.categoryAxis.labels.dy = -2
        bc.categoryAxis.labels.angle = 30
        bc.categoryAxis.categoryNames = list(value.keys())
        drawing.add(bc)
        story.append(drawing)

    # drawing = Drawing((PAGE_WIDTH - inch * 2) / 2, 0)
    # data = [
    #     (13, 5, 20, 22, 37, 45, 19, 4),
    # ]
    # bc = VerticalBarChart()
    # bc.x = (PAGE_WIDTH - inch * 2) / 2 + 20
    # bc.y = 30
    # bc.height = 160
    # bc.width = (PAGE_WIDTH - inch * 2) / 2 - 20
    # bc.data = data
    # bc.strokeColor = None
    # bc.bars[0].fillColor = colors.indianred
    # bc.valueAxis.valueMin = 0
    # bc.valueAxis.valueMax = 50
    # bc.valueAxis.valueStep = 10
    # bc.categoryAxis.labels.boxAnchor = 'ne'
    # bc.categoryAxis.labels.dx = 8
    # bc.categoryAxis.labels.dy = -2
    # bc.categoryAxis.labels.angle = 30
    # bc.categoryAxis.categoryNames = ['Jan-99', 'Feb-99', 'Mar-99',
    #                                  'Apr-99', 'May-99', 'Jun-99', 'Jul-99', 'Aug-99']
    # drawing.add(bc)
    # story.append(drawing)

    story.append(Spacer(width=0, height=30))
    story.append(
        Paragraph("2. Molecular Property Interval Statistics", style=styles["Heading2"])
    )
    story.append(table)
    doc.build(story, onFirstPage=s_myFirstPage, onLaterPages=myLaterPages)


def s_download(request):
    if request.method == "POST":
        filename = request.POST.get("filename")
        pdf_filepath = (
            os.path.join(settings.SITE_ROOT, "static/files/s_pdf/") + filename + ".pdf"
        )
        if not os.path.exists(pdf_filepath):
            gen_s_pdf(filename)
        buffer = open(pdf_filepath, "rb")
        # return HttpResponse('antin')
        return FileResponse(
            buffer, as_attachment=True, filename=pdf_filepath.split("/")[-1]
        )
    else:
        return HttpResponseRedirect(reverse("home:index"))


def gen_pdf2(file_data, pdf_filepath, filename, index):
    ori_mol = file_data.iloc[0]
    gen_mol = file_data.iloc[int(index)]
    doc = SimpleDocTemplate(pdf_filepath, pagesize=A4)
    doc.title = "Detail Information"
    doc.smiles = ori_mol["smiles"]
    story = []
    story.append(Spacer(width=0, height=30))
    file1 = (
        os.path.join(settings.SITE_ROOT, "static/files/orimol_img/")
        + filename
        + "_"
        + str(index)
        + ".png"
    )
    file2 = (
        os.path.join(settings.SITE_ROOT, "static/files/genmol_img/")
        + filename
        + "_"
        + str(index)
        + ".png"
    )
    file3 = (
        os.path.join(settings.SITE_ROOT, "static/files/trans_img/")
        + filename
        + "_"
        + str(index)
        + ".png"
    )
    if not os.path.exists(file1):
        gen_molimg(smiles=ori_mol["smiles"], path=file1)
    if not os.path.exists(file2):
        gen_molimg(smiles=gen_mol["smiles"], path=file2)
    if not os.path.exists(file3):
        gen_transimg(smarts=gen_mol["transformation"], path=file3)
    ori_info = [
        get_Para("Original Molecule", header_style),
        get_Para(ori_mol["smiles"], value_style),
        Image(file1, width=120, height=120, hAlign="CENTER"),
    ]
    trans_info = [
        get_Para("Transformation", header_style),
        get_Para(gen_mol["transformation"], value_style),
        Image(file3, width=140, height=70, hAlign="CENTER"),
    ]
    gen_info = [
        get_Para("Generated Molecule", header_style),
        get_Para(gen_mol["smiles"], value_style),
        Image(file2, width=120, height=120, hAlign="CENTER"),
    ]
    info = [ori_info, trans_info, gen_info]
    table = Table(
        info,
        spaceBefore=2,
        colWidths=[doc.width * 0.26, doc.width * 0.4, doc.width * 0.34],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )
    story.append(Spacer(width=0, height=20))
    story.append(Paragraph("1. Transformation information", style=styles["Heading2"]))
    story.append(table)

    story.append(Spacer(width=0, height=30))
    story.append(Paragraph("2. Property information", style=styles["Heading2"]))
    # Part 1
    physicochemical_value = [
        ori_mol[item].round(3) for item in ["LogS", "LogP", "LogD"]
    ]
    physicochemical_gen_value = [
        gen_mol[item].round(3) for item in ["LogS", "LogP", "LogD"]
    ]
    physicochemical_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("O_V", header_center_style),
            get_Para("G_V", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    physicochemical = np.array(
        [
            physicochemical_property,
            physicochemical_value,
            physicochemical_gen_value,
            physicochemical_comment,
        ]
    ).T
    physicochemical = np.vstack((physicochemical_header, physicochemical)).tolist()

    # Part 2
    absorption_value = [
        ori_mol[item]
        for item in ["Caco-2", "MDCK", "Pgp-inh", "Pgp-sub", "HIA", "F(20%)", "F(30%)"]
    ]
    round_absorption_value = [
        ori_mol[item].round(3)
        for item in ["Caco-2", "MDCK", "Pgp-inh", "Pgp-sub", "HIA", "F(20%)", "F(30%)"]
    ]
    absorption_value_gen = [
        gen_mol[item]
        for item in ["Caco-2", "MDCK", "Pgp-inh", "Pgp-sub", "HIA", "F(20%)", "F(30%)"]
    ]
    round_absorption_value_gen = [
        gen_mol[item].round(3)
        for item in ["Caco-2", "MDCK", "Pgp-inh", "Pgp-sub", "HIA", "F(20%)", "F(30%)"]
    ]
    absorption_decision = get_absor_decision(absorption_value_gen)
    absorption_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("O_V", header_center_style),
            get_Para("G_V", header_center_style),
            get_Para("Decision", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    absorption = np.array(
        [
            absorption_property,
            round_absorption_value,
            round_absorption_value_gen,
            absorption_decision,
            absorption_comment,
        ]
    ).T
    absorption = np.vstack((absorption_header, absorption)).tolist()

    # Part 3
    distribution_value = [
        ori_mol[item].round(3) for item in ["PPB", "VDss", "BBB", "Fu"]
    ]
    distribution_value_gen = [
        gen_mol[item].round(3) for item in ["PPB", "VDss", "BBB", "Fu"]
    ]
    distribution_decision = get_dis_decision(distribution_value_gen)
    distribution_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("O_V", header_center_style),
            get_Para("G_V", header_center_style),
            get_Para("Decision", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    distribution = np.array(
        [
            distribution_property,
            distribution_value,
            distribution_value_gen,
            distribution_decision,
            distribution_comment,
        ]
    ).T
    distribution = np.vstack((distribution_header, distribution)).tolist()

    # Part 4
    metabolism_value = [
        ori_mol[item].round(3)
        for item in [
            "CYP1A2-inh",
            "CYP1A2-sub",
            "CYP2C19-inh",
            "CYP2C19-sub",
            "CYP2C9-inh",
            "CYP2C9-sub",
            "CYP2D6-inh",
            "CYP2D6-sub",
            "CYP3A4-inh",
            "CYP3A4-sub",
        ]
    ]
    metabolism_value_gen = [
        gen_mol[item].round(3)
        for item in [
            "CYP1A2-inh",
            "CYP1A2-sub",
            "CYP2C19-inh",
            "CYP2C19-sub",
            "CYP2C9-inh",
            "CYP2C9-sub",
            "CYP2D6-inh",
            "CYP2D6-sub",
            "CYP3A4-inh",
            "CYP3A4-sub",
        ]
    ]
    metabolism_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("O_V", header_center_style),
            get_Para("G_V", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    metabolism = np.array(
        [
            metabolism_property,
            metabolism_value,
            metabolism_value_gen,
            metabolism_comment,
        ]
    ).T
    metabolism = np.vstack((metabolism_header, metabolism)).tolist()

    # Part 5
    excretion_value = [ori_mol[item].round(3) for item in ["CL", "T12"]]
    excretion_value_gen = [gen_mol[item].round(3) for item in ["CL", "T12"]]
    excretion_decision = get_excret_decision(excretion_value_gen)
    excretion_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("O_V", header_center_style),
            get_Para("G_V", header_center_style),
            get_Para("Decision", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    excretion = np.array(
        [
            excretion_property,
            excretion_value,
            excretion_value_gen,
            excretion_decision,
            excretion_comment,
        ]
    ).T
    excretion = np.vstack((excretion_header, excretion)).tolist()

    # Part 7
    toxicity_value = [
        ori_mol[item].round(3)
        for item in [
            "hERG",
            "H-HT",
            "DILI",
            "Ames",
            "ROA",
            "FDAMDD",
            "SkinSen",
            "Carcinogenicity",
            "EC",
            "EI",
            "Respiratory",
        ]
    ]
    toxicity_value_gen = [
        gen_mol[item].round(3)
        for item in [
            "hERG",
            "H-HT",
            "DILI",
            "Ames",
            "ROA",
            "FDAMDD",
            "SkinSen",
            "Carcinogenicity",
            "EC",
            "EI",
            "Respiratory",
        ]
    ]
    toxicity_decision = get_toxicity_decision(toxicity_value_gen)
    toxicity_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("O_V", header_center_style),
            get_Para("G_V", header_center_style),
            get_Para("Decision", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    toxicity = np.array(
        [
            toxicity_property,
            toxicity_value,
            toxicity_value_gen,
            toxicity_decision,
            toxicity_comment,
        ]
    ).T
    toxicity = np.vstack((toxicity_header, toxicity)).tolist()

    # Part 8
    env_value = [ori_mol[item].round(3) for item in ["BCF", "IGC50", "LC50", "LC50DM"]]
    env_value_gen = [
        gen_mol[item].round(3) for item in ["BCF", "IGC50", "LC50", "LC50DM"]
    ]
    env_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("O_V", header_center_style),
            get_Para("G_V", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    env = np.array([env_property, env_value, env_value_gen, env_comment]).T
    env = np.vstack((env_header, env)).tolist()

    # Part 9
    pathway_value = [
        ori_mol[item].round(3)
        for item in [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ]
    ]
    pathway_value_gen = [
        gen_mol[item].round(3)
        for item in [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ]
    ]
    pathway_decision = get_pathway_decision(pathway_value_gen)
    pathway_header = np.array(
        [
            get_Para("Property", header_style),
            get_Para("O_V", header_center_style),
            get_Para("G_V", header_center_style),
            get_Para("Decision", header_center_style),
            get_Para("Comment", header_style),
        ]
    )
    pathway = np.array(
        [
            pathway_property,
            pathway_value,
            pathway_value_gen,
            pathway_decision,
            pathway_comment,
        ]
    ).T
    pathway = np.vstack((pathway_header, pathway)).tolist()

    physicochemical_table = Table(
        physicochemical,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.18,
            doc.width * 0.1,
            doc.width * 0.1,
            doc.width * 0.62,
        ],
    )
    physicochemical_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (3, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )

    absorption_table = Table(
        absorption,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.18,
            doc.width * 0.1,
            doc.width * 0.1,
            doc.width * 0.14,
            doc.width * 0.48,
        ],
    )
    absorption_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (3, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )

    distribution_table = Table(
        distribution,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.18,
            doc.width * 0.1,
            doc.width * 0.1,
            doc.width * 0.14,
            doc.width * 0.48,
        ],
    )
    distribution_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )

    metabolism_table = Table(
        metabolism,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.18,
            doc.width * 0.1,
            doc.width * 0.1,
            doc.width * 0.62,
        ],
    )
    metabolism_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )
    excretion_table = Table(
        excretion,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.18,
            doc.width * 0.1,
            doc.width * 0.1,
            doc.width * 0.14,
            doc.width * 0.48,
        ],
    )
    excretion_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )
    toxicity_table = Table(
        toxicity,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.18,
            doc.width * 0.1,
            doc.width * 0.1,
            doc.width * 0.14,
            doc.width * 0.48,
        ],
    )
    toxicity_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )

    env_table = Table(
        env,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.18,
            doc.width * 0.1,
            doc.width * 0.1,
            doc.width * 0.62,
        ],
    )
    env_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )

    pathway_table = Table(
        pathway,
        spaceBefore=2,
        colWidths=[
            doc.width * 0.18,
            doc.width * 0.1,
            doc.width * 0.1,
            doc.width * 0.14,
            doc.width * 0.48,
        ],
    )
    pathway_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )
    )

    story.append(Paragraph("2.1 Physicochemical Property", style=styles["Heading4"]))
    story.append(physicochemical_table)
    story.append(Paragraph("2.2 Absorption", style=styles["Heading4"]))
    story.append(absorption_table)
    story.append(Paragraph("2.3 Distribution", style=styles["Heading4"]))
    story.append(distribution_table)
    story.append(Paragraph("2.4 Metabolism", style=styles["Heading4"]))
    story.append(metabolism_table)
    story.append(Paragraph("2.5 Excretion", style=styles["Heading4"]))
    story.append(excretion_table)
    story.append(Paragraph("2.6 Toxicity", style=styles["Heading4"]))
    story.append(toxicity_table)
    story.append(Paragraph("2.7 Environmental toxicity", style=styles["Heading4"]))
    story.append(env_table)
    story.append(Paragraph("2.8 Tox21 pathway", style=styles["Heading4"]))
    story.append(pathway_table)
    doc.build(story, onFirstPage=myFirstPage, onLaterPages=myLaterPages)


def d_pdf(request, filename, index):
    pdf_filepath = (
        os.path.join(settings.SITE_ROOT, "static/files/d_pdf/")
        + filename
        + "_"
        + str(index)
        + ".pdf"
    )
    if not os.path.exists(pdf_filepath):
        file_data = pd.read_csv(
            os.path.join(settings.SITE_ROOT, "static/files/result/tmp/")
            + filename
            + ".csv"
        )
        # data = pd.read_csv(file_data)
        gen_pdf2(file_data, pdf_filepath, filename, index)
    buffer = open(pdf_filepath, "rb")
    return FileResponse(
        buffer, as_attachment=True, filename=pdf_filepath.split("/")[-1]
    )
