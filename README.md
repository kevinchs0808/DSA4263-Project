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

### Incident Exposure Analysis

Firstly, EDA was performed based on the existing literature about fraudster’s modus operandi on exposure theory, whereby the fraudsters seek to maximise their exposure to keep themselves clean under the radar to reduce their chance of getting caught. One set of factors that might have significance in fraud detection was the exposure of the case, i.e., how “well-known” is the incident, to both the authorities and the general public. Hence, investigation into the type of the authorities contacted for fraudulent insurance claims and the number of witnesses present for the insurance claims were performed.

In our dataset, two columns, “witnesses” and “authorities contacted”, represent the exposure of the event in the above dimensions. 

<p align="center">
  <img src="./images/Fraud_Reported_By_Witness.png" width=560/>
  <img src="./images/Fraud_Reported_By_Authorities_Contacted.png" width=500/>
</p>

Based on the visualisations provided above, surprisingly, both types of cases exhibit similar distributions regarding the number of witnesses and the authorities_contacted column. This suggests a potential collusion between fraudsters and members of the public to procure witnesses, and possibly deceitful behaviour towards authorities to manipulate their involvement in the incident. It's plausible that such actions stem from past fraud detection methods that heavily relied on exposure, prompting fraudsters to pay special attention to these aspects and take extra precautions to conceal their actions. Consequently, we hypothesise that the public exposure of an incident may not reliably predict insurance fraud.

### Fraudster Incentive Analysis

Secondly, the report delved into the mindset of the fraudsters’ psychology through fraudster incentive. In the context of fraudulent claims, it is important to ask cui bono? If the fraudster does not benefit from the act, they would not have the incentive to commit insurance fraud in the first place. Therefore, the severity of the incident was scrutinised closely for this second EDA.

<p align="center">
  <img src="./images/Fraud_Reported_By_Incident_Severity.png"  width=470/>
  <img src="./images/Fraud_Reported_And_Proportion_By_Authorities_Contacted.png"  width=610/>
</p>

As observed from the visualisations above, the majority of Major Damage reported are fraudulent. This trend is highlighted in particular by the left figure, where the absolute number of fraud cases, as well as the proportion of fraud cases in this class of incident_severity. 
As such, we hypothesise that incident severity could be a significant predictor in fraud detection, where cases reported as Major Damage are more likely to be fraudulent. This presumption is grounded in economic motivations, as fraudsters are more inclined to label their cases as major for the larger insurance payouts. Given the substantial monetary investment and logistical efforts involved in orchestrating fraudulent schemes, fraudsters seek commensurate economic gains to offset their expenses effectively.

### Anomaly Analysis

Lastly, EDA was used to discover anomalies in the dataset. After conducting exhaustive search on all the variables in the dataset, a noticeably large number of insurance claims in which the cilents who indicated “chess' or “cross-fit' under insured_hobbies column are fraudulent as shown in the visualisation below.

<p align="center">
  <img src="./images/Anomaly_Analysis.png" />
</p>

As correlation does not imply causation, it is preposterous to conclude that insurance clients who declared “chess” and “cross-fit” are more likely to commit fraudulent claims. The high frequencies in the “chess” and “cross-fit” are attributed towards cultural factors and location-driven. Since the dataset used is based in the USA and the culture in the USA is arguably dominantly anglo-saxon, the popularity sentiments towards cross-fit in the USA and chess as part of common pastimes will be reflected in this dataset unsurprisingly.. 

As such, the report made the following three hypotheses. Firstly, the car brand and the insured claimants policy affects the fraudulent claims. Secondly, either exposure theory or fraudster incentive has an influence on the fraudulent claims. Lastly, the hobbies “chess” and “cross-fit” do not contribute to the fraudulent claims outcome.


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

