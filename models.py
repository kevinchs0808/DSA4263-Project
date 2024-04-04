from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from imblearn.over_sampling import SMOTENC, RandomOverSampler, ADASYN

import shap
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import lime

import parameters

################################################################################
### Oversampling Helper Functions

def perform_SMOTENC_oversampling(X, y):
    """
    Perform oversampling using SMOTENC.

    Parameters:
    X (DataFrame): The input features.
    y (Series): The target variable.

    Returns:
    DataFrame, Series: The resampled features and target variable.
    """
    # Define the list of numerical columns
    num_columns_list = [
        'months_as_customer',
        'age',
        'policy_deductable',
        'policy_annual_premium',
        'umbrella_limit',
        'capital-gains',
        'capital-loss',
        'incident_hour_of_the_day',
        'total_claim_amount',
        'injury_claim',
        'property_claim',
        'vehicle_claim'
    ]

    # Obtain the indices of categorical columns
    cat_columns_indices = [i for i, col in enumerate(X.columns) if col not in num_columns_list]

    # Initialize SMOTENC with categorical feature indices and sampling strategy
    smtnc = SMOTENC(
        categorical_features=cat_columns_indices,
        sampling_strategy='not majority',
        random_state=42
    )

    # Perform oversampling
    X_balanced, y_balanced = smtnc.fit_resample(X, y)

    return X_balanced, y_balanced

def perform_random_oversampling(X, y):
    """
    Perform oversampling using RandomOverSampler.

    Parameters:
    X (DataFrame): The input features.
    y (Series): The target variable.

    Returns:
    DataFrame, Series: The resampled features and target variable.
    """
    # sampling_strategy, increase minority count to match the majority count
    ros = RandomOverSampler(sampling_strategy = 'not majority', random_state=42)
    X_balanced, y_balanced = ros.fit_resample(X, y)
    return X_balanced, y_balanced

def perform_ADASYN(X, y):
    """
    Perform oversampling using ADASYN.

    Parameters:
    X (DataFrame): The input features.
    y (Series): The target variable.

    Returns:pos_label=Y
    DataFrame, Series: The resampled features and target variable.
    """
    # sampling_strategy, increase minority count to match the majority count
    ada = ADASYN(sampling_strategy = 'not majority', random_state=42)
    X_balanced, y_balanced = ada.fit_resample(X, y)
    return X_balanced, y_balanced

def imbalanced_dataset_treatment(X, y, oversampling_strategy):
    """
    Handle imbalanced dataset by applying the specified oversampling strategy.

    Parameters:
    X (DataFrame): The input features.
    y (Series): The target variable.
    oversampling_strategy (str): The oversampling strategy to use. Options are "SMOTENC", "RandomOverSampler", "ADASYN", or "None" (no oversampling).

    Returns:
    DataFrame, Series: The resampled features and target variable.

    Raises:
    ValueError: If an unknown oversampling strategy is passed.
    """
    if oversampling_strategy == "SMOTENC":
        X_balanced, y_balanced = perform_SMOTENC_oversampling(X, y)
    elif oversampling_strategy == "RandomOverSampler":
        X_balanced, y_balanced = perform_random_oversampling(X, y)
    elif oversampling_strategy == "ADASYN":
        X_balanced, y_balanced = perform_ADASYN(X, y)
    elif oversampling_strategy == "None":
        return X, y
    else:
        raise ValueError(f"Unknown oversampling strategy: {oversampling_strategy}")

    return X_balanced, y_balanced

################################################################################
### K Fold Cross Validation Helper Functions

def get_scoring_function(metric):
    """
    Get the scoring function based on the provided metric.

    Parameters:
    metric: The evaluation metric to compute ('accuracy', 'precision', 'recall', 'f1-score', 'roc', 'pr_auc').

    Returns:
    The corresponding scoring function.
    """
    if metric == 'accuracy':
        return accuracy_score
    elif metric == 'precision':
        return precision_score
    elif metric == 'recall':
        return recall_score
    elif metric == 'f1-score':
        return f1_score
    elif metric == 'roc':
        return roc_auc_score
    elif metric == 'pr_auc':
        return average_precision_score
    else:
        raise ValueError("Invalid metric. Choose from 'accuracy', 'precision', 'recall', 'f1-score', 'roc', 'pr_auc'.")


def perform_stratified_k_fold(model, X, y, k, oversampling_strategy, metric):
    """
    Perform stratified cross-validation and return the scores.

    Parameters:
    model : object
        The sklearn model that should be used.
    X : array-like of shape (n_samples, n_features)
        The feature matrix.
    y : array-like of shape (n_samples,)
        The target vector.
    k : int, default=5
        The number of folds for cross-validation.
    oversampling_strategy : str
        The strategy for oversampling.
    metric : {'accuracy', 'precision', 'recall', 'f1-score', 'roc_auc', 'pr_auc'}
        The evaluation metric to compute.

    Returns:
    float
        The mean score from all folds of the cross-validation.
    """
    skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
    scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Perform oversampling on the X_train and y_train dataframe
        X_train_bal, y_train_bal = imbalanced_dataset_treatment(X_train, y_train,
                                                          oversampling_strategy)

        # Fit the model to the X_train dataset
        model.fit(X_train_bal, y_train_bal)

        # Obtain the prediction for the
        y_pred = model.predict(X_test)

        # Compute the evaluation metric using the scoring function
        score_function = get_scoring_function(metric)
        if metric in ['f1-score', 'recall', 'precision']:
            score = score_function(y_test, y_pred, pos_label=1)
        else:
            score = score_function(y_test, y_pred)
        scores.append(score)

    cv_score = sum(scores) / len(scores)
    return cv_score

################################################################################
### Modelling Classes

def merge_dicts(*dicts):
    '''
    This helper function merges dictionaries together.
    '''
    return {k: v for d in dicts for k, v in d.items()}

class IndividualModel:

    # This is a class that will be used to create a single model.
    # Please ensure that your function supports sci-kit learn interface.
    
    def __init__(
            self, model_func, param_info, 
            X_train, X_test, y_train, y_test, 
            tuned_params = {}, static_params = {}
            ):

        '''
        This is the constructor for the IndividualModel class. 
        Each IndividualModel object will represent a single model warpper function around a sci-kit learn model.

        Parameters:

        model_func: function call to create the model, NOT THE MODEL ITSELF
        param_info: dictionary that contains information used to finetune the model
        X_train: training data, this data is not balanced
        X_test: testing data, this data is not balanced and will not be balanced.
        y_train: training labels, this data is not balanced
        y_test: testing labels, this data is not balanced and will not be balanced. 
        tuned_params: dictionary with the tuned hyperparameters for the model, if applicable. Else is empty
        other_params: dictionary with parameters that are not hyperparameters, such as random_state.
        '''

        self.model_func = model_func

        # initialise a model
        all_params = merge_dicts(tuned_params, static_params)
        self.model = model_func(**all_params)

        self.tuned_params = tuned_params
        self.static_params = static_params

        # param_info is a dictionary that contains the potential hyperparameters, cross-validation number, Optuna direction, and number of trials
        # they are used to finetune the model
        self.param_info = param_info

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self, oversampling_strategy = 'SMOTENC'):
        '''
        This method trains model by doing the following:
        1. Oversample the training data with the specified method
        2. Fit the model with the oversampled data
        '''

        X_train_bal, y_train_bal = imbalanced_dataset_treatment(
            self.X_train, self.y_train, oversampling_strategy
            )
        # This is to train the model
        self.model.fit(X_train_bal, y_train_bal)

    def predict(self):
        # This is to generate predictions using a trained model
        self.y_pred = self.model.predict(self.X_test)

    def train_predict(self, oversampling_strategy = 'SMOTENC'):
        # This is to train and predict in one go
        self.train(oversampling_strategy)
        self.predict()

    def finetune(self, oversampling_strategy = 'SMOTENC', metric = "f1-score", **kwargs):
        '''
        This function finetunes an individual model using Optuna, and update the model accordingly

        parameters:
        oversample_method: method to oversample the data. Default is SMOTENC
        metric: metric to be used to evaluate the model. Default is F1 Score
        kwargs: additional parameters that can be passed to the optuna.create_study() func
        '''

        if not len(self.tuned_params) == 0:
            print("Model has already been finetuned. Existing hyperparameters will discarded.")

        def objective(trial):

            testing_params = {}

            param_candidates = self.param_info['potential_hyperparameters']
            
            # Loop through the potential hyperparameters and suggested values
            for param, requirement in param_candidates.items():

                if requirement['finetune'] == True:
                    if requirement['trial'] == 'categorical':
                        testing_params[param] = trial.suggest_categorical(
                            name=param, 
                            choices=requirement['choices']
                        )

                    elif requirement['trial'] == 'int':
                        testing_params[param] = trial.suggest_int(
                            name=param, 
                            low=requirement['low_value'], 
                            high=requirement['high_value'], 
                            log=requirement['use_log'],
                            step=requirement['finetuning_step']
                        )
                    elif requirement['trial'] == 'float':
                        testing_params[param] = trial.suggest_float(
                            name=param, 
                            low=requirement['low_value'], 
                            high=requirement['high_value'], 
                            log=requirement['use_log'],
                            step=requirement['finetuning_step']
                        )
                else:
                    testing_params[param] = requirement['exact_value']
        
            model = self.model_func(**testing_params)
            
            cv_score = perform_stratified_k_fold(
                model, self.X_train, self.y_train, 
                self.param_info['tuning_options']['cv_number'], 
                oversampling_strategy, metric
                )
            return cv_score
        
        tuning_options = self.param_info['tuning_options']
        
        # Create new optuna study
        study = optuna.create_study(direction = tuning_options["optuna_direction"])
        study.optimize(objective, n_trials = tuning_options["n_trials"])

        chosen_trial = study.best_trial

        self.tuned_params = chosen_trial.params

        self.validation_score = chosen_trial.value

        # Update the model for future usage

        all_params = merge_dicts(self.tuned_params, self.static_params)
        self.model = self.model_func(**all_params)

        return self.validation_score, self.tuned_params

    def evaluate(self):
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred, pos_label=1)
        self.recall = recall_score(self.y_test, self.y_pred, pos_label=1)
        self.f1_score = f1_score(self.y_test, self.y_pred, pos_label=1)

        self.performance_score = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score
        }

        return self.performance_score
    
    def plot_confusion_matrix(self, save_path=None):
        # Generate confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)

        # Plot confusion matrix as heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False, ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix for {self.param_info["model_name"]}')

        # Save the figure to the specified path if provided
        if save_path:
            fig.savefig(save_path)

        return fig

    def shap_explanation(self, is_tree=False, class_to_observe=0):
        shap.initjs()

        # Create the explainer
        explainer = shap.Explainer(self.model)

        #shap_values = explainer.shap_values(self.X_test)

        #return shap.summary_plot(shap_values, self.X_test)

        shap_values = explainer(self.X_test)

        if is_tree:
            return shap.plots.beeswarm(shap_values[:,:,class_to_observe])
        else:
            return shap.plots.beeswarm(shap_values)
    
    def lime_explanation(self, chosen_index, num_features):

        # how do you get chosen_instance?

        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names = self.X_train.columns,
            class_names=self.y_train.unique().tolist(),
            kernel_width=parameters.LIME_KERNEL_WIDTH
        )

        predict_probability = lambda x: self.model.predict_proba(x).astype(float)

        exp = explainer.explain_instance(self.X_test.values[chosen_index], predict_probability, num_features=num_features)

        return exp.show_in_notebook(show_all=True)
    

# class ModellingPipeline:

#     def __init__(self, models, X_train, X_test, y_train, y_test):

#         self.models = {}

#         for model in models:

#             self.models[model['model_name']] = IndividualModel(model, X_train, X_test, y_train, y_test)

#     def tune_all_models(self):

#         for model_name in self.models.keys():

#             validation_score, chosen_hyperparameters = self.models[model_name].finetune()
#             self.models[model_name].train()
#             self.models[model_name].predict()
#             performance_score = self.models[model_name].evaluate()

#             print(f"{model_name} finished")
#             print(f"Validation Score: {validation_score} across {self.models[model_name].information['cv_number']} iterations of cross-validation")
#             print(f"Chosen Hyperparameters: {chosen_hyperparameters}")
#             print(f"Performance Score: {performance_score}")
    
#     def check_performance(self, model_name):
#         return self.models[model_name].performance_score
    
#     def plot_confusion_matrix(self, model_name, save_path = None):
#         self.models[model_name].plot_confusion_matrix(save_path=save_path)
    
#     def shap_explanation(self, model_name, chosen_index):
#         self.models[model_name].shap_explanation(chosen_index)
    
#     def lime_explanation(self, model_name, chosen_index, num_features):
#         self.models[model_name].lime_explanation(chosen_index, num_features)

