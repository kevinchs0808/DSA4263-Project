from sklearn.model_selection import cross_val_score
import optuna
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import lime

import parameters

class BaselineModel:

    def __init__(self, information, X_train, X_test, y_train, y_test):
        self.information = information
        self.hyperparameters = {}
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def finetuning(self):
        def objective(trial):

            testing_hyperparameters = {}

            for key, requirement in self.information['potential_hyperparameters'].items():
                if requirement['finetune'] == True:
                    if requirement['trial'] == 'categorical':
                        testing_hyperparameters[key] = trial.suggest_categorical(
                            name=key, 
                            choices=requirement['choices']
                        )

                    elif requirement['trial'] == 'int':
                        testing_hyperparameters[key] = trial.suggest_int(
                            name=key, 
                            low=requirement['low_value'], 
                            high=requirement['high_value'], 
                            log=requirement['use_log'],
                            step=requirement['finetuning_step']
                        )
                    elif requirement['trial'] == 'float':
                        testing_hyperparameters[key] = trial.suggest_float(
                            name=key, 
                            low=requirement['low_value'], 
                            high=requirement['high_value'], 
                            log=requirement['use_log'],
                            step=requirement['finetuning_step']
                        )
                else:
                    testing_hyperparameters[key] = requirement['exact_value']
        
            model = self.information['model'](**testing_hyperparameters)
            score = cross_val_score(estimator=model, X=self.X_train, y=self.y_train, cv=self.information['cv_number']).mean()
            return score
        
        study = optuna.create_study(direction = self.information["optuna_direction"])
        study.optimize(objective, n_trials = self.information["n_trials"])

        trial = study.best_trial

        self.hyperparameters = trial.params

        self.validation_score = trial.value

        self.model = self.information['model'](**self.hyperparameters)

        return self.validation_score, self.hyperparameters

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def evaluate(self):
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred, pos_label='Y')
        self.recall = recall_score(self.y_test, self.y_pred, pos_label='Y')
        self.f1_score = f1_score(self.y_test, self.y_pred, pos_label='Y')

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
        ax.set_title(f'Confusion Matrix for {self.information["model_name"]}')

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

            self.models[model['model_name']] = BaselineModel(model, X_train, X_test, y_train, y_test)

    def execute(self):

        for model_name in self.models.keys():

            validation_score, chosen_hyperparameters = self.models[model_name].finetuning()
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

