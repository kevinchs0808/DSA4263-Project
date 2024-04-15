# DSA4263 - Insurance Claim Fraud Detection

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Insights](#insights)
    - [Incident Exposure Analysis](#incident-exposure-analysis)
    - [Fraudster Incentive Analysis](#fraudster-incentive-analysis)
    - [Anomaly Analysis](#anomaly-analysis)
- [Usage](#usage)
  - [Install Resources](#install-resources)
- [ML Product Solution](#ml-product-solution)
  - [Download Datasets](#download-datasets)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Training Models](#training-models)
    - [Initialisation](#initialisation)
    - [Baseline Training](#baseline-training)
    - [Optuna Hyperparameter Tuning](#optuna-hyperparameter-tuning)
    - [Results and Confusion Matrix Generation](#results-and-confusion-matrix-generation)
    - [ROC Plot Generation](#roc-plot-generation)
    - [SHAP Plot Generation](#shap-plot-generation)
    - [Export Best Model](#export-best-model)
- [Contributors](#contributors)
- [References](#references)
- [Software License](#software-license)

## Problem Statement

<p align="center">
  <img src="./images/Insurance_Industry.png" />
</p>

In the insurance industry, clients normally make annual payments in the hope of getting some monetary claims in case some unexpected events happen such as property destruction, car accidents and illness.

However, there are pockets of malicious individuals who apply for fraudulent claims in hope of getting payment. This creates a lose-lose situation for everyone: Longer processing time for insurance professionals; larger insurance premiums for cilents. 

Therefore, we have the two problem statements for this project.

- Firstly, generate insights from the insurance dataset.
- Secondly, provide a ML product solution to automate the claims

## Project Structure

This is the project structure for our repository

<p align="center">
  <img src="./images/project_structure.png" />
</p>

## Data Description

- The dataset contains 1000 rows and 40 columns.
- The columns recorded consist of the clients' details and their respective claims.
- The dataset is sampled from various insurance companies from three US states: South Carolina, Virginia and Albany.
- Each row represents a single anonymised claim record submitted by the insurance client to the insurance companies.
- `fraud_reported` label indicates if the claim is fraudulent or not.
- There are 200 fraudulent claims present in this dataset.
    - For this dataset, LightGBM with ADASYN oversampling is the best model.
    - More details about the model performance could be found in the `final_report.pdf`.
- More details about each column belonging to policy and claims can be found in https://data.mendeley.com/datasets/992mh7dk9y/2 (Aqqad, 2023).

## Insights

### Incident Exposure Analysis

<p align="center">
  <img src="./images/Fraud_Reported_By_Witness.png" width=560/>
  <img src="./images/Fraud_Reported_By_Authorities_Contacted.png" width=500/>
</p>

- Fraudulents claims are invariant of the number of witness and authorities contacted. This might imply Fraudsters might try to use authorities or witness to increase legitimacy of their claims.

### Fraudster Incentive Analysis

<p align="center">
  <img src="./images/Fraud_Reported_And_Proportion_By_Authorities_Contacted.png"  width=610/>
</p>

- Majority of the Major Damage reported are fraudulent. Fraudster might prefer to file Major Damage due to the incentive present.

### Anomaly Analysis

<p align="center">
  <img src="./images/Anomaly_Analysis.png" />
</p>

- Interestingly, majority of the claims in which the insurance claimants who declared `Chess` and `Cross-fit` are fraudulent.

## Usage

### Install Resources
First, change to home directory.
```
cd ~
```
To install the resources, use `git clone` to clone the GitHub repository into your machine.
```
git clone git clone https://github.com/kevinchs0808/DSA4263-Project.git
cd DSA4263-Project
```
Next, install pip and the required packages using the following code:
```
sudo apt install python3-pip
pip install -r requirements.txt
```

## ML Product Solution

### Download Datasets
The dataset is added to the `Data\Raw` directory for the this repository. This step is used if the latest dataset is needed.

Go to your shell (Git Bash, Powershell etc.).

First, change your directory to `Data`.
```
cd Data
```
If you are not at `DSA4263-Project` directory, use:
```
cd ~/DSA4263-Project/Data
```
Then, change your directory to `Raw`.
```
cd Raw
```
Then, download the insurance datasets using the following code
```
curl -o dataset.zip "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/992mh7dk9y-2.zip"
```
Then, Unzip the file and extract its contents
```
unzip dataset.zip
```
Then, remove the zip file
```
rm dataset.zip
```
### Preprocessing Pipeline
Once the dataset's acqusition is completed, use this code to go back to main directory of the repository
```
cd ..
cd ..
```
You may go to `main.ipynb`

Read the dataset
```
df = pd.read_csv("Data/Raw/insurance_claims.csv")
```
Remark:
- If you encountered the `FileNotFoundError: [Errno 2] No such file or directory: 'Data/Raw/insurance_claims.csv'` in the directory, return back to the `Download Datasets` section above.

If you are using tree-based (Decision Tree or Random Tree) or gradient-boosting-tree models (XGBoost or LightGBM), normalisation is not needed. 

Use this pipeline in the notebook with `normalization` switched to `False`.
```
X_train, X_test, y_train, y_test = preprocessing.preprocess_pipeline(
    df, encoding=True, normalization=True)
```

If you are using Neural Network (Multi-Layer Perceptron or Tensorflow) or Linear models (Logistic Regression or Support Vector Machine), normalisation is needed.

Use this pipeline in the notebook with `normalization` switched to `TRUE`.
```
X_train, X_test, y_train, y_test = preprocessing.preprocess_pipeline(
    df, encoding=True, normalization=True)
```

### Training Models
Our product machine learning model provides the following two models in Object-Oriented-Implementation from `model.py`.

- Baseline model for quick model generation with no Hyperparameter tuning.
- Tuned model with Optuna Bayesian Search Hyperparameter tuning.

#### Initialisation

We will use the LightGBM as a demonstration in this documentation.

We will import the libraries first
```
import lightgbm as lgb
```
We will call the preprocess_pipeline function
```
X_train_lgb, X_test_lgb, y_train_lgb, y_test_lgb = preprocessing.preprocess_pipeline(
    df, encoding = True, normalization = False)
```
We will initialise the Lightgbm object
```
lgb_static_params = {
    "random_state": 42,
    "verbose": -1,
}

lgb_model = models.IndividualModel(
    model_func=lgb.LGBMClassifier,
    param_info=parameters.LGBM_INFORMATION,
    X_train=X_train_lgb,
    X_test=X_test_lgb,
    y_train=y_train_lgb,
    y_test=y_test_lgb,
    static_params=lgb_static_params
)
```
#### Baseline Training

These are the following `oversampling_strategy` available for baseline model generation:
- `SMOTENC`
- `RandomOverSampler`
- `ADASYN`
- `None` (no oversampling strategy is ued)

Suppose the baseline model used SMOTENC as the default oversampling strategy.

There are two options in baseline Training.

##### Option 1

To train the baseline model
```
lgb_model.train(oversampling_strategy="SMOTENC", baseline=True)
```
To predict on the test dataset
```
lgb_model.predict(baseline=True)
```
To evaluate the lightgbm results
```
lgb_model.evaluate(baseline=True)
```

##### Option 2

To call these functions in one shot.
```
lgb_model.train_predict_eval(oversampling_strategy="SMOTENC", baseline=True)
```

#### Optuna Hyperparameter Tuning

These are the following `oversampling_strategy` options available for the Optuna's hyperparameter tuning:
- `SMOTENC` (Default)
- `RandomOverSampler`
- `ADASYN`
- `None` (no oversampling strategy is used)

These are the following `metric` options available for the Optuna's hyperparameter tuning:
- `accuracy`
- `precision`
- `recall`
- `f1-score` (Default)
- `roc`
- `pr_auc`

Optuna will choose the best model with highest 5-Fold Stratified Cross Validation Score for the choosen metric using Bayesian Optimisation Search.

Suppose you want you fine tune lightgbm with `oversampling_strategy = 'SMOTENC'` and `metric = 'f1-score'`. Use this code
```
lgb_model.finetune(oversampling_strategy="SMOTENC", `metric = 'f1-score')
```
Upon completion, it will return the following output.
```
(0.710178901268986,
 {'n_estimators': 100,
  'num_leaves': 148,
  'learning_rate': 0.1485023100776579,
  'subsample': 0.8500000000000001,
  'colsample_bytree': 0.7500000000000001,
  'min_child_samples': 11,
  'reg_alpha': 1.5072102857938577e-07,
  'reg_lambda': 2.651919424678569e-05})
```
#### Results and Confusion Matrix Generation

##### Baseline

To predict on the test dataset with your baseline lightgbm
```
lgb_model.predict(baseline=True)
```
To evaluate the lightgbm baseline model results
```
lgb_model.evaluate(baseline=True)
```
To obtain the confusion matrix
```
cm_lgb_baseline = lgb_model.plot_confusion_matrix()
```
##### Tuned

To predict on the test dataset with your baseline lightgbm
```
lgb_model.predict(baseline=False)
```
To evaluate the lightgbm baseline model results
```
lgb_model.evaluate(baseline=False)
```
Remark: 
- If you encountered `ValueError("Model has not been finetuned yet. Please finetune the model first.")`, please call the `finetune` function found in Optuna Hyperparameter Tuning section before calling this function.

To obtain the confusion matrix
```
cm_lgb_tuned = lgb_model.plot_confusion_matrix()
```
#### ROC Plot Generation

##### Baseline
To obtain the baseline plot
```
auc_lgb_base = lgb_model.plot_auc(baseline=True)
```
##### Tuned
```
auc_lgb_tuned = lgb_model.plot_auc(baseline=False)
```
Remark: 
- If you encountered `ValueError("Model has not been finetuned yet. Please finetune the model first.")`, please call the `finetune` function found in Optuna Hyperparameter Tuning section before calling this function.
#### SHAP Plot Generation
Our product used the SHAP (SHapley Additive exPlanations) to explain the model.

##### Baseline
To plot baseline LightGBM SHAP's plot
```
lgb_model.shap_explanation(baseline = True)
```
##### Tuned
To plot tuned LightGBM SHAP's plot
```
lgb_model.shap_explanation(baseline = False)
```
Remark: 
- If you encountered `ValueError("Model has not been finetuned yet. Please finetune the model first.")`, please call the `finetune` function found in Optuna Hyperparameter Tuning section before calling this function.
#### Export Best Model
You have the option to export either Baseline or Tuned model in the `pkl` format. The shortlisted model
will be stored in the `model` folder.

##### Baseline
To export the baseline LightGBM model to the model folder
```
model_path = "models/lgb_baseline.pkl"
lgb_model_best.export_model(path=model_path)
```

##### Tuned
To export the baseline LightGBM model to the model folder
```
model_path = "models/lgb_tuned.pkl"
lgb_model_best.export_model(path=model_path)
```
Remark: 

- If you encountered `ValueError("Model has not been finetuned yet. Please finetune the model first.")`, please call the `finetune` function found in Optuna Hyperparameter Tuning section before calling this function.

- If the `export_model` is called the second time, The model will be updated to a new one with this message `Tuned model already exists. Overwriting the tuned model.`


## Contributors
- Kevin Christian
- Loo Guan Yee
- Sun Peizhi
- Vivek Bagai

## References
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining, 2623–2631. https://doi.org/10.1145/3292500.3330701
- Aqqad, A. (2023). insurance_claims. Mendeley Data, 2. https://doi.org/10.17632/992mh7dk9y.2 (This repository's dataset)  
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (n.d.). SMOTE: synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16(1), 321–357. https://doi.org/https://dl.acm.org/doi/10.5555/1622407.1622416 
- Chen, T., & Guestrin, C. (2016). XGBoost. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/2939672.2939785 
- He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). Adasyn: Adaptive Synthetic Sampling Approach for imbalanced learning. 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence). https://doi.org/10.1109/ijcnn.2008.4633969 
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., & Ma, W. (2017). LightGBM: a highly efficient gradient boosting decision tree. NIPS’17: Proceedings of the 31st International Conference on Neural Information Processing Systems, 3149–3157. https://doi.org/https://dl.acm.org/doi/10.5555/3294996.3295074 
- Lundberg, S., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NIPS 2017. https://doi.org/10.48550/arXiv.1705.07874 
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. The Journal of Machine Learning Research, 12, 2825–2830. https://doi.org/https://dl.acm.org/doi/10.5555/1953048.2078195 
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. https://doi.org/10.48550/arXiv.1206.2944 

## Software License
This project is licensed under the [MIT License](LICENSE).