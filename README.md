# DSA4263 - Insurance Claim Fraud Detection

## Problem Statement

In the insurance industry, clients normally make annual payments in the hope of getting some monetary claims in case some unexpected events happen such as property destruction, car accidents, as well as heart attacks. 

<p align="center">
  <img src="./images/Insurance_Industry.png" />
</p>

Unfortunately, some of these claims are misused for personal advantage, causing many insurance companies to take more time in manual claim processing as higher due diligence is required. However, some cases may require fast decision making especially when there’s a patient whose condition has become critical and needs some surgery to be done as soon as possible. With manual processing, multiple issues can occur due to this situation, which include:

- Unhappy customers can be caused by the following cases:
    - Death because of not being able to pay for surgery or treatment
    - Lose their house and cars
- Huge loss for the company
- Slow claim processing
- Loss of Productivity due to manual processing

## Project Structure

<p align="center">
  <img src="./images/project_structure.png" />
</p>

## Insights


## Challenges

As the dataset is imbalanced, it may affect the performance of our models. Hence, it is important to use several oversampling strategy as follows:
- SMOTENC: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html
- Random Oversampling: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
- ADASYN: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html

Additionally, there may be some features which are not relevant. As there are 40 features in the dataset, each of us in a team of 4 started by analyzing 10 features and see if they are relevant.

## Solution

We are using several models to build the insurance claim fraud detection system, which include the following:
- Decision Tree: https://scikit-learn.org/stable/modules/tree.html
- Random Forest:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- XGBoost: https://xgboost.readthedocs.io/en/stable/
- LightGBM: https://lightgbm.readthedocs.io/en/stable/
- Multi Perceptron Layer: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
- SVM: https://scikit-learn.org/stable/modules/svm.html
- Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

For this project, we aim to maximize the F1-Score of the models, as we want to make sure that all of the historical cases are predicted correctly, but at the same time the models can predict correctly as well!

In the end, other than having great performing models, we would also love them to be interpratable as well.

Hence we also show the SHAP visualization based on the feature importance for each model.

This part will allow the business team to understand what factors are likely to be associated with fraudulent cases.

## Result

