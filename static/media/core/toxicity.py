'''
Description: Calculate the toxicity properties of molecule
Author: Kotori Y
Date: 2020-10-31 12:06:51
LastEditors: Kotori Y
LastEditTime: 2020-11-06 09:09:24
FilePath: \admetMesh\admetMesh\admetEvaluation\toxicity.py
AuthorMail: kotori@cbdd.me
'''

import os
import sys

# sys.path.append('..')
# from models.predictor import pred_admet
from rdkit import Chem
from .admetConfig import SmartDir
from .substructure_filter import SubstrutureFilter
import pandas as pd
from collections.abc import Iterable


class Toxicity:

    def __init__(self, mols):

        self.mols = mols if isinstance(mols, Iterable) else (mols,)
        self.smis = None

    def ScreenToxicityFragments(self):
        res = pd.DataFrame()
        atoms = pd.DataFrame()
        files = os.listdir(SmartDir)

        for file in files:
            endpoint = os.path.splitext(file)[0]
            if endpoint != "PAINS":
                for x in SubstrutureFilter().screening(self.mols, endpoint):
                    length = 0 if x["MatchedNames"] == ['-'] else len(x["MatchedNames"])
                    res[f"{endpoint}"] = [length]
                    # print(x["MatchedAtoms"][0])
                    atoms[f"{endpoint}"] = [x["MatchedAtoms"][0]]
        return res, atoms

    def CalculateAllToxicityProperties(self):
        # T = self.CalculateToxicityProperties()
        # T21 = self.CalculateToxicity21Properties()
        Tsub = self.ScreenToxicityFragments()

        # return T, T21, Tsub
        return Tsub


if "__main__" == __name__:
    import pandas as pd

    smi = 'CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC'
    mol = Chem.MolFromSmiles(smi)

    mols = [mol] * 10
    T = Toxicity(mols)
    res, atom = T.ScreenToxicityFragments()
    print(res)
    print(atom)
    # print(Tsub.columns)
