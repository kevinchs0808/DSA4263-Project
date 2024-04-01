# List of Categorical Columns

cate_cols = [
    'policy_state', 'insured_sex',
    'insured_education_level', 'insured_occupation', 'insured_hobbies', 'insured_relationship',
    'incident_type', 'collision_type', 'incident_severity',
    'authorities_contacted', 'incident_state', 
    'property_damage', 'police_report_available',
    'auto_make', 'auto_region', 'auto_type'
]

RANDOM_FOREST_INFORMATION = {
    'model_name': 'Random Forest',
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
    'tuning_options': {
        'cv_number': 3,
        'optuna_direction': "maximize",
        "n_trials": 100
    }
}

XGB_INFORMATION = {
    'model_name': 'XGBoost',
    'potential_hyperparameters': {
        "enable_categorical": {
            "finetune": False,
            "exact_value": True,
        },
        'n_estimators': {
            'finetune': True,
            'low_value': 10,
            'high_value': 500,
            'exact_value': None,  # This is optional, you can specify an exact value if needed
            'trial': 'int',
            'use_log': True,
            'finetuning_step': 1
        },
        'max_depth': {
            'finetune': True,
            'low_value': 3,
            'high_value': 36,
            'exact_value': None,
            'trial': 'int',
            'use_log': False, 
            'finetuning_step': 1
        },
        'learning_rate': {
            'finetune': True,
            'low_value': 0.01,
            'high_value': 0.5,
            'exact_value': None,
            'trial': 'float', 
            'use_log': False,
            'finetuning_step': 0.01
        },
        'subsample': {
            'finetune': True,
            'low_value': 0.2,
            'high_value': 1.0,
            'exact_value': None,
            'trial': 'float',
            'use_log': False,
            'finetuning_step': 0.05
        },
        'colsample_bytree': {
            'finetune': True,
            'low_value': 0.2,
            'high_value': 1.0,
            'exact_value': None,
            'trial': 'float',
            'use_log': False,
            'finetuning_step': 0.05
        },
        'gamma': {
            'finetune': True,
            'low_value': 0,
            'high_value': 5,
            'exact_value': None,
            'trial': 'float',
            'use_log': False,
            'finetuning_step': 0.1
        },
        'reg_alpha': {
            'finetune': True,
            'low_value': 0,
            'high_value': 5,
            'exact_value': None,
            'trial': 'float',
            'use_log': False,
            'finetuning_step': 0.1
        },
        'reg_lambda': {
            'finetune': True,
            'low_value': 0,
            'high_value': 5,
            'exact_value': None,
            'trial': 'float',
            'use_log': False,
            'finetuning_step': 0.1
        }
    },
    'tuning_options': {
        'cv_number': 5,
        'optuna_direction': "maximize",
        "n_trials": 100
    }
}

LIME_KERNEL_WIDTH = 5