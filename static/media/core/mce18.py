'''
Description: Calculate MCE-18 index based on paper \
    "Are We Opening the Door to a New Era of Medicinal Chemistry or Being Collapsed to a Chemical Singularity"
Author: Kotori Y
Date: 2020-11-28 16:40:13
LastEditors: Kotori Y
LastEditTime: 2020-11-29 12:33:54
FilePath: \MCE-18\mce18.py
AuthorMail: kotori@cbdd.me
'''

from rdkit.Chem import AllChem as Chem
from collections import Counter
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdMolDescriptors import CalcNumSpiroAtoms


class MCE18:
    """calculate the descriptor MCE-18, which can effectively 
    score molecules by novelty in terms of their cumulative sp3 complexity.
    """

    def __init__(self, mol):
        """Init

        Parameters
        ----------
        mol : rdkit.rdchem.Mol
            molecule to be calculated
        """
        self.mol = mol
        self.nC = len(
            [atom for atom in mol.GetAtoms()
                if atom.GetAtomicNum() == 6]
        )

    def _MolMatchSmarts(self, mol, smarts):
        """*internal only*
        """
        patt = Chem.MolFromSmarts(smarts)
        res = mol.GetSubstructMatches(patt)
        return res

    def CalculateQ1Index(self):
        """calculate normalized quadratic index (Q1 index)

        Returns
        -------
        Q1Index : float
            normalized quadratic index

        Reference
        ---------
        Balaban, Theor Chem Acc
        doi: 10.1007/BF00555695

        """
        matrix = GetAdjacencyMatrix(self.mol)
        M = sum(matrix.sum(axis=1) ** 2)
        N = self.mol.GetNumAtoms()
        Q1Index = 3 - 2 * N + M / 2
        return Q1Index

    def CalculateAR(self):
        """check the presence of an aromatic or heteroaromatic ring

        Returns
        -------
        AR : int, 0 or 1
            the presence of an aromatic or heteroaromatic ring (0 or 1)
        """
        smarts = "a"
        AR = bool(self._MolMatchSmarts(self.mol, smarts))
        AR = int(AR)
        return AR

    def CalculateNAR(self):
        """check the presence of an aliphatic or a heteroaliphatic ring

        Returns
        -------
        NAR : int, 0 or 1
            the presence of an aliphatic or a heteroaliphatic ring
        """
        smarts = "[A;R]"
        NAR = bool(self._MolMatchSmarts(self.mol, smarts))
        NAR = int(NAR)
        return NAR

    def CalculateCHIRAL(self):
        """check the presence of a chiral center (0 or 1)

        Returns
        -------
        CHIRAL : int, 0 or 1
            the presence of a chiral center (0 or 1)
        """
        CHIRAL = (Chem.CalcNumAtomStereoCenters(self.mol)) > 0
        CHIRAL = int(CHIRAL)
        return CHIRAL

    def CalculateSPIRO(self):
        """check the presence of a spiro center (0 or 1)

        Returns
        -------
        SPIRO : int, 0 or 1
            the presence of a spiro center (0 or 1)
        """
        SPIRO = CalcNumSpiroAtoms(self.mol) > 0
        SPIRO = int(SPIRO)
        return SPIRO

    def CalculateSP3(self):
        """calculate the portion of sp3-hybridized carbon atoms (from 0 to 1)

        Returns
        -------
        sp3 : float, from 0 to 1
            the portion of sp3-hybridized carbon atoms (from 0 to 1)
        """
        smarts = "[CX4]"
        sp3 = len(self._MolMatchSmarts(self.mol, smarts))
        sp3 = sp3/self.nC
        return sp3

    def CalculateCyc(self):
        """calculate the portion of cyclic carbons that 
        are sp3 hybridized (from 0 to 1)

        Returns
        -------
        Cyc : float, from 0 to 1
            the portion of cyclic carbons that are sp3 hybridized (from 0 to 1)
        """
        smarts = "[CX4;R]"
        Cyc = len(self._MolMatchSmarts(self.mol, smarts))
        Cyc = Cyc/self.nC
        return Cyc

    def CalculateAcyc(self):
        """calculate the portion of acyclic carbon atoms that are sp3 hybridized (from 0 to 1)

        Returns
        -------
        Acyc : float, from 0 to 1
            the portion of acyclic carbon atoms that are sp3 hybridized (from 0 to 1)
        """
        smarts = "[CX4;R0]"
        Acyc = len(self._MolMatchSmarts(self.mol, smarts))
        Acyc = Acyc/self.nC
        return Acyc
    
    def CalculateMCE18(self):
        
        AR = self.CalculateAR()
        NAR = self.CalculateNAR()
        CHIRAL = self.CalculateCHIRAL()
        SPIRO = self.CalculateSPIRO()
        sp3 = self.CalculateSP3()
        Cyc = self.CalculateCyc()
        Acyc = self.CalculateAcyc()
        Q1 = self.CalculateQ1Index()

        part1 = AR + NAR + CHIRAL + SPIRO
        part2 = sp3 + Cyc - Acyc
        part3= 1 + sp3
        part4 = part1 + (part2 / part3)
        mce = part4 * Q1
        
        return mce



if "__main__" == __name__:

    # smiles = "C1NCCN(C2=CC3N(CC)C=C(C(=O)O)C(=O)C=3C=C2F)C1"
    smiles = "C1=CC(C(CC)(C/C=C/C2=CC=CC=C2)N(C)CC2CC2)=CC=C1"
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)

    demo = MCE18(mol)

    print("DONE")
