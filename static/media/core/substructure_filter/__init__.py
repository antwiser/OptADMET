'''
Description: 
Author: Kotori Y
Date: 2020-10-24 16:03:49
LastEditors: Kotori Y
LastEditTime: 2020-11-04 15:23:56
FilePath: \admetMesh\admetMesh\substructure_filter\__init__.py
AuthorMail: kotori@cbdd.me
'''

from collections.abc import Iterable
from functools import partial

try:
    from check_substructure import CheckWithSmarts
except Exception:
    from .check_substructure import CheckWithSmarts


class SubstrutureFilter:

    def __init__(self):
        pass

    def screening(self, mols, endpoint):
        func = partial(CheckWithSmarts, endpoint=endpoint)
        mols = mols if isinstance(mols, Iterable) else (mols,)
        res = list(func(mols))
        return res


if '__main__' == __name__:
    from rdkit import Chem
    import pandas as pd

    smis = [
        'OC1=C[C-]2[OH+]C(c3ccc(O)c(O)c3)=C(O)C=C2C(O)=C1',
        'O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12',
        'OCC(O)C(O)C(O)C(O)CO',
    ]
    mols = [Chem.MolFromSmiles(smi) for smi in smis]

    Filter = SubstrutureFilter()
    res = Filter.screening(mols, endpoint='Acute_Aquatic_Toxicity')
    print(res)
