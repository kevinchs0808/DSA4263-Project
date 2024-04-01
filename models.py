from sklearn.model_selection import cross_val_score
import optuna
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import lime

import parameters

def merge_dicts(*dicts):
    '''
    This function merges dictionaries together.
    '''
    return {k: v for d in dicts for k, v in d.items()}

class IndividualModel:

    # This is a class that will be used to create a single model.
    # Please ensure that your function supports sci-kit learn interface.
    
    def __init__(self, model_func, param_info, X_train, X_test, y_train, y_test, tuned_params = {}, static_params = {}):

        '''
        This is the constructor for the IndividualModel class. 
        Each IndividualModel object will represent a single model warpper function around a sci-kit learn model.

        Parameters:

        model_func: function call to create the model, NOT THE MODEL ITSELF
        param_info: dictionary that contains information used to finetune the model
        X_train: training data
        X_test: testing data
        y_train: training labels
        y_test: testing labels
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

    def train(self):
        # This is to train the model
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        # This is to generate predictions using a trained model
        self.y_pred = self.model.predict(self.X_test)

    def train_predict(self):
        # This is to train and predict in one go
        self.train()
        self.predict()

    def finetune(self, **kwargs):
        '''
        This function finetunes an individual model using Optuna, and update the model accordingly

        parameters:

        kwargs: additional parameters that can be passed to the optuna.create_study() func
        '''

        if not len(self.tuned_params) == 0:
            print("Model has already been finetuned. Existing hyperparameters will discarded.")

        def objective(trial):

            testing_hyperparameters = {}

            param_candidates = self.param_info['potential_hyperparameters']
            
            # Loop through the potential hyperparameters and suggested values
            for param, requirement in param_candidates.items():

                if requirement['finetune'] == True:
                    if requirement['trial'] == 'categorical':
                        testing_hyperparameters[param] = trial.suggest_categorical(
                            name=param, 
                            choices=requirement['choices']
                        )

                    elif requirement['trial'] == 'int':
                        testing_hyperparameters[param] = trial.suggest_int(
                            name=param, 
                            low=requirement['low_value'], 
                            high=requirement['high_value'], 
                            log=requirement['use_log'],
                            step=requirement['finetuning_step']
                        )
                    elif requirement['trial'] == 'float':
                        testing_hyperparameters[param] = trial.suggest_float(
                            name=param, 
                            low=requirement['low_value'], 
                            high=requirement['high_value'], 
                            log=requirement['use_log'],
                            step=requirement['finetuning_step']
                        )
                else:
                    testing_hyperparameters[param] = requirement['exact_value']
        
            model = self.model_func(**testing_hyperparameters)
            score = cross_val_score(
                estimator=model, X=self.X_train, y=self.y_train, 
                cv=self.param_info['tuning_options']['cv_number'],
                ).mean()
            return score
        
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

    def shap_explanation(self, chosen_index):
        shap.initjs()

        # Create the explainer
        explainer = shap.Explainer(self.model)

        shap_values = explainer.shap_values(self.X_test)

        return shap.summary_plot(shap_values, self.X_test)
    
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
    
class ModellingPipeline:

    def __init__(self, models, X_train, X_test, y_train, y_test):

        self.models = {}

        for model in models:

            self.models[model['model_name']] = IndividualModel(model, X_train, X_test, y_train, y_test)

    def tune_all_models(self):

        for model_name in self.models.keys():

            validation_score, chosen_hyperparameters = self.models[model_name].finetune()
            self.models[model_name].train()
            self.models[model_name].predict()
            performance_score = self.models[model_name].evaluate()

            print(f"{model_name} finished")
            print(f"Validation Score: {validation_score} across {self.models[model_name].information['cv_number']} iterations of cross-validation")
            print(f"Chosen Hyperparameters: {chosen_hyperparameters}")
            print(f"Performance Score: {performance_score}")
    
    def check_performance(self, model_name):
        return self.models[model_name].performance_score
    
    def plot_confusion_matrix(self, model_name, save_path = None):
        self.models[model_name].plot_confusion_matrix(save_path=save_path)
    
    def shap_explanation(self, model_name, chosen_index):
        self.models[model_name].shap_explanation(chosen_index)
    
    def lime_explanation(self, model_name, chosen_index, num_features):
        self.models[model_name].lime_explanation(chosen_index, num_features)

