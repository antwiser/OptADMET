'''
Description: 
Author: Kotori Y
Date: 2020-11-03 15:25:15
LastEditors: Kotori Y
LastEditTime: 2020-11-30 14:33:22
FilePath: \admetMesh\admetMesh\admetEvaluation\medicinalchemistry.py
AuthorMail: kotori@cbdd.me
'''

import os
import sys
# sys.path.append("..")
from .substructure_filter import SubstrutureFilter

import pandas as pd
import numpy as np
from rdkit.Chem.Descriptors import qed
from rdkit import RDConfig
from collections.abc import Iterable
from functools import partial

sys.path.append(RDConfig.RDContribDir)
from SA_Score import sascorer
from NP_Score import npscorer

from .mce18 import MCE18


ContriDir = RDConfig.RDContribDir
filename = os.path.join(ContriDir, 'NP_Score/publicnp.model.gz')
fscore = npscorer.pickle.load(npscorer.gzip.open(filename))

ruleList = {
    # "Egan": {"TPSA": [0, 132], "LogP": [-1, 6]},
    # "Veber": {"nRot": [0, 10], "TPSA": [0, 140], "nHB": [0, 12]},
    "Lipinski": {"MW": [0, 500], "LogP": [None, 5],
                 "nHD": [0, 5], "nHA": [0, 10]},
    "Pfizer": {"LogP": [3, None], "TPSA": [0, 75]},
    "GSK": {"MW": [0, 400], "LogP": [None, 4]},
    # "Oprea": {"nRing": [3, None], "nRig": [18, None], "nRot": [6, None]},
    # "REOS": {"MW": [200, 500], "LogP": [-5, 5], "nHD": [0, 5],
    #          "nHA": [0, 10], "nRot": [0, 8], "TPSA": [0, 150],
    #          "fChar": [-4, 4]},
    "GoldenTriangle": {"MW": [200, 500], "LogD": [-2, 5]},
    # "Xu": {"nHD": [0, 5], "nHA": [0, 10], "nRot": [3, 35],
    #        "nRing": [1, 7], "nHev": [10, 50]},
    # "Ro4": {"MW": [0, 400], "LogP": [None, 4], "nHD": [0, 4],
    #         "nHA": [0, 8], "TPSA": [0, 120]},
    # ##########################################
    # "bRo5": {"MW": [0, 1000], "LogP": [-2, 10], "nHD": [0, 6],
    #          "nHA": [0, 15], "TPSA": [0, 250], "nRot": [0, 20]},
    # "OralMacrocycles": {"MW": [0, 1000], "LogP": [None, 10],
    #                     "nHD": [0, 5], "TPSA": [0, 250]},
}


class MedicinalChemistry(SubstrutureFilter):

    def __init__(self, mols):
        super().__init__()
        self.mols = mols if isinstance(mols, Iterable) else (mols, )

    def qed_score(self):
        """
        Quantitative Drug Likeness (QED)

        :return: score
        """
        score = list(map(qed, self.mols))
        return score

    
    def CalculateSAscore(self):
        """
        A function to estimate ease of synthesis (synthetic accessibility) of drug-like molecules
        --->SAscore

        Reference:
            (1) `Ertl Peter (2009)`_.

        :return: ease of synthesis
        :rtype: float

        .. _Ertl Peter (2009):
            https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8

        """
        return list(map(sascorer.calculateScore, self.mols))

    def CalculateNPscore(self):
        """
        A function to calculate the natural product-likeness score
        --->NPscore
        
        Reference:
            (1) `Ertl Peter (2008)`_.
        
        :return: product-likeness score
        :rtype: list
        
        .. _Ertl Peter (2008):
            https://pubs.acs.org/doi/abs/10.1021/ci700286x
        
        """
        func = partial(npscorer.scoreMol, fscore=fscore)
        return list(map(func, self.mols))

    def CalculateSP3(self):
        
        fsp3 = []
        mce = []
        for mol in self.mols:
            app = MCE18(mol)
            fsp3.append(app.CalculateSP3())
            mce.append(app.CalculateMCE18())
        return fsp3, mce

    def _CheckDruglikenessRule(
            self,
            limited={
                "MW": [0, 500],
                "LogP": [None, 500],
                "nHD": [0, 5],
                "nHA": [0, 10],
            },
            **kwgrs):

        allowed = kwgrs.get("allowed")
        res = pd.DataFrame()
        flag = 0
        for key, vals in limited.items():
            prop = kwgrs.get(key)

            bo = ((vals[0] is None) or (prop >= vals[0])) \
                & ((vals[1] is None) or (prop <= vals[1]))
            
            status = np.where(bo, "Accepted", "Rejected")
            res[key] = list(zip(prop, status))
            flag += ~bo
            
        pfizer = kwgrs.get('pfizer')
        # print(pfizer)
        if not pfizer:
            res["Disposed"] = np.where(flag < allowed, "Accepted", "Rejected")
        else:
            res["Disposed"] = np.where(flag < allowed, "Rejected", "Accepted")
        return res

    def ScreeningMed(self):
        res = pd.DataFrame()
        atoms = pd.DataFrame()
        
        endpoints = ["Alarm_NMR", "BMS", "Chelating", "PAINS"]
        
        for endpoint in endpoints:
            # endpoint = os.path.splitext(file)[0]
            # print(endpoint)
            length = []
            a = []
            for x in super().screening(self.mols, endpoint):
                leng = 0 if x["MatchedNames"] == ['-'] else len(x["MatchedNames"])
                length.append(leng)
                a.append(x["MatchedAtoms"])
            res[f"{endpoint}"] = length
            atoms[f"{endpoint}"] = a
            
        return res, atoms

    def CalculatedMedicinalChemistryProperties(self, **kwgrs):

        fsp3, mce = self.CalculateSP3()

        frag, fragAtoms = self.ScreeningMed()
        Med = pd.DataFrame({
            "QED": self.qed_score(),
            "Synth": self.CalculateSAscore(),
            "Fsp3": fsp3,
            "MCE-18": mce,
            "Natural Product-likeness": self.CalculateNPscore(),
        })
        Med = pd.concat([Med, frag], axis=1)
        # print(Med)

        ruleRes = {}
        for name, rule in ruleList.items():
            allowed = 2 if name == "Lipinski" else 1
            pfizer = (name == "Pfizer")
            res = self._CheckDruglikenessRule(rule, allowed=allowed, pfizer=pfizer, **kwgrs)
            ruleRes[name] = res
            Med[name] = res["Disposed"].values
            # out[f"{name}Detail"] = res
        
        return Med, fragAtoms


if "__main__" == __name__:
    from rdkit import Chem

    smis = pd.read_csv("../data/example.txt", header=None)
    smis = np.array(smis).flatten()
    
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    demo = MedicinalChemistry(mols)
    
    np.random.seed(777)
    props = {
        "MW": np.random.randint(100, 800, 100),
        "LogP": 3 * np.random.randn(100) + 0.2,
        "LogD": 3 * np.random.randn(100),
        "nHA": np.random.randint(0, 10, 100),
        "nHD": np.random.randint(0, 10, 100),
        "TPSA": np.random.randint(0, 200, 100),
    }
    props["nHB"] = props["nHA"] + props["nHD"]
    
    Med, fragAtoms = demo.CalculatedMedicinalChemistryProperties(**props)
    print(Med, end='\n\n')
    print(fragAtoms)
