# Interpretable dynamic directed graph convolutional network（IDDGCN） on mutation-driven drug reponse prediction

`IDDGCN` uses directed graphs to distinguish between sensitivity and resistance relationships, and dynamically updating the weights of these relationships to reflect the specific interactions of different nodes. We also applied interpretability models to our prediction framework and proposed a method for constructing ground truth based on drug response relationships for evaluation. The prediction experimental results demonstrate that IDDGCN outperforms existing state-of-the-art models, exhibiting excellent predictive power and generalization ability, providing a new approach for precision oncology and targeted drug development.

## Table of Contents
- [Framework](#framework)
- [System Requirements](#system-requirements)
- [Datasets](#datasets)
- [Run IDDGCN](#run-iddgcn)
- [Acknowledgements](#acknowledgements)
- [References](#References)

## Framework
![Framework](image/流程图.png)

## System Requirements
The source code developed in Python 3.7 using tensorflow  2.7.0. The required python dependencies are given below. IDDGCN is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.
```markdown
tensorflow>=2.7.0
scipy>=1.7.3
scikit-learn>= 0.22.1
numpy>= 1.21.6
pandas>=1.3.5
h5py>=3.8.0
openpyxl>=3.1.2
PubChemPy>=1.0.4
```

## Datasets
The `datasets` directory contains all the experimental data used in IDDGCN. [MetaKB](https://search.cancervariants.org/#*)[^1].
In and folders, we have full data with two genes-specific test sets, which are used for model generalization and intragenic experiments. Each subfolder contains the complete data used for the experiments:
- `datasets/data_new_ABL1` Contains new mutation-drug response data for ABL1 used for generalization experiments.
- `datasets/data_new_KRAS` Contains new mutation-drug response data for KRAS used for generalization experiments.
- `datasets/data_single_ABL1` Contains single gene mutation-drug response data for ABL1 used for intragenic experiments.
- `datasets/data_single_KRAS` Contains single gene mutation-drug response data for KRAS used for intragenic experiments.

## Run IDDGCN

To train IDDGCN and evaluate results, follow these steps:

1. **Training**: Run `IDDGCN.py` in the `prediction_code` folder to train the IDDGCN model. You can run this step to train the model or directly proceed to step two to see the evaluation results.
```markdown
python prediction_code/IDDGCN.py
```
2. **Evaluation**: Run `IDDGCN_eval.py` in the `prediction_code` folder to evaluate the prediction results.
```markdown
python prediction_code/IDDGCN_eval.py
```
3. **Ground Truth Construction**: You can use `ground_truth` in the `explanation` folder to construct ground truth, or directly use the pre-constructed `gt_all.csv` in the `explanation/data` folder.
4. **Explanation**: Directly run `GnnExplainer` and `explaiNE` in the `explanation` folder to explain the prediction results.
```markdown
python explanation/GnnExplainer.py
python explanation/explaiNE.py
```
5. **Evaluation of Explanation Models**: Run `eval_test.py` in the `explanation` folder to evaluate the explanation models.  
```markdown
python explanation/eval_test.py
```
## Acknowledgements

## References
[^1]:Wagner A H, Walsh B, Mayfield G, et al. A harmonized meta-knowledgebase of clinical interpretations of somatic genomic variants in cancer[J]. Nature genetics, 2020, 52(4): 448-457.
