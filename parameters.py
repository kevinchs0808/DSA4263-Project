from sklearn.ensemble import RandomForestClassifier

RANDOM_FOREST_INFORMATION = {
    'model': RandomForestClassifier,
    'model_name': 'Random_Forest',
    'potential_hyperparameters': {
        'n_estimators': {
            'finetune': True,
            'low_value': 10,
            'high_value': 30,
            'exact_value': 20,
            'trial': 'int',
            'use_log': True,
            'finetuning_step': 1
        },
        'max_depth': {
            'finetune': True,
            'low_value': 5,
            'high_value': 10,
            'exact_value': 8,
            'trial': 'int',
            'use_log': True,
            'finetuning_step': 1
        },
        'criterion': {
            'finetune': True,
            'choices': ['gini', 'entropy', 'log_loss'],
            'exact_value': 'gini',
            'trial': 'categorical'
        }
    },
    'cv_number': 3,
    'optuna_direction': "maximize",
    "n_trials": 100
}

LIME_KERNEL_WIDTH = 5