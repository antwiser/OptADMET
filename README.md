# OptADMET

<div id="top" align="center">
  
  [![logo](https://cadd.nscc-tj.cn/deploy/optadmet/static/home/img/index.png)](https://cadd.nscc-tj.cn/deploy/optadmet/)

  OptADMET: Chemical transformation webserver for ADMET Optimization
  
  [![GitHub Repo stars](https://img.shields.io/github/stars/antwiser/OptADMET?style=social)](https://github.com/antwiser/OptADMET/stargazers)
  [![Tutorial](https://img.shields.io/badge/Tutorial-passing-green)](https://cadd.nscc-tj.cn/deploy/optadmet/tutorial/)
  [![WebSite](https://img.shields.io/badge/WebSite-blue)](https://cadd.nscc-tj.cn/deploy/optadmet/)

</div>

---

# Overview
  
One of the great challenges in drug discovery is rationalizing lead optimization. In this phase, compounds that have been identified as the potential drug candidates need to be modified carefully with appropriate absorption, distribution, metabolism, excretion, and toxicity (ADMET) properties. The key questions faced by medicinal chemists are which compound(s) should be made next and how to balance multiple properties. To avoid the poor or ill-advised decisions, it is crucial to draw credible transformation rules from multiple experimental assays and apply them for efficient optimization. Therefore, OptADMET, the first integrated chemical transformation rule platform covering 32 important ADMET properties, is able to provide multiple-property rules and apply invaluable experience for lead optimization. For multiple-property rule database, a total of 177,191 experimental data are used for analysis, which have produced 41,779 credible transformation rules. Besides, 239,194 accurate predicted molecular data are integrated with the initial data for expanded structural conversion exploration, which have converted 146,450 rules as the supplement. Based on the large and credible rule database, OptADMET is able to find desirable substructure transformation and guide the efficient multi-parameter optimization for the queried molecule. Additionally, to benefit the final decision, the ADMET profile of all optimized molecules will also be provided by OptADMET for comprehensive evaluation.

![Figure 1](https://cadd.nscc-tj.cn/deploy/optadmet/static/home/img/tutorial_overview.png)

Fig 1. The constitute of OptADMET webserver

# Usage

## 1. Create environment

```
conda create --name <env> --file requirements.txt
```

Important libraries supported in the environment include: 📢

- Django == 2.2
- DGL == 0.7.2
- PyTorch == 1.6.0
- RDKit == 2020.09.1.0
- reportlab == 3.6.8
- sqlparse == 0.4.1
- scikit-learn == 1.0.1

## 2. Download the necessary datasets

