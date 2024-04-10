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
            'low_value': 100, 
            'high_value': 500,
            'exact_value': 100, # sk learn default is 100 for random forest
            'trial': 'int',
            'use_log': False,
            'finetuning_step': 50
        },
        'max_depth': {
            'finetune': True,
            'low_value': 1,
            'high_value': 10,
            'exact_value': 2,
            'trial': 'int',
            'use_log': True,
            'finetuning_step': 1
        },
        'criterion': {
            'finetune': True,
            'choices': ['gini', 'entropy', 'log_loss'],
            'exact_value': 'gini', # default is gini
            'trial': 'categorical'
        },
        'min_samples_split': { # minimum samples to split a leaf node
            'finetune': True,
            'low_value': 2,
            'high_value': 10,
            'exact_value': 2, # default is 2
            'trial': 'int',
            'use_log': False,
            'finetuning_step': 1
        },
        'min_samples_leaf': { # minimum samples in leaf node
            'finetune': True,
            'low_value': 1,
            'high_value': 10,
            'exact_value': 1, # default is 1
            'trial': 'int',
            'use_log': False,
            'finetuning_step': 1
        },
        'max_features': { # Number of features to conisder while looking for best split
            'finetune': True,                  # Whether to fine-tune this parameter
            'choices': ['sqrt', 'log2', None], # Available choices for the parameter
            'exact_value': 'sqrt',             # Default or exact value for the parameter
            'trial': 'categorical'             # Type of tuning trial (e.g., 'categorical')
        },
        'bootstrap': { # bootstrap samples were used to build the tree
            'finetune': True,
            'choices': [True, False],
            'exact_value': True, # Default is True
            'trial': 'categorical'
        },
        'random_state': {
            'finetune': False,
            'low_value': 1,
            'high_value': 100,
            'exact_value': 42,
            'trial': 'int'
        }
    },
    'tuning_options': {
        'cv_number': 5,
        'optuna_direction': "maximize",
        "n_trials": 100
    }
}

DECISION_TREE_INFORMATION = {
    'model_name': 'Decision Tree',
    'potential_hyperparameters': {
        'max_depth': {
            'finetune': True,
            'low_value': 1,
            'high_value': 10,
            'exact_value': 2,
            'trial': 'int',
            'use_log': True,
            'finetuning_step': 1
        },
        'min_samples_split': {
            'finetune': True,
            'low_value': 2,
            'high_value': 10,
            'exact_value': 2,
            'trial': 'int',
            'use_log': True,
            'finetuning_step': 1
        },
        'min_samples_leaf': {
            'finetune': True,
            'low_value': 1,
            'high_value': 5,
            'exact_value': 1,
            'trial': 'int',
            'use_log': True,
            'finetuning_step': 1
        },
        'random_state': {
            'finetune': False,
            'low_value': 1,
            'high_value': 100,
            'exact_value': 42,
            'trial': 'int'
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
        },
        'random_state': {
            'finetune': False,  # Set to False since we are not fine-tuning this parameter
            'exact_value': 42,  # Specify '42' as the exact value
    }
    },

    'tuning_options': {
        'cv_number': 5,
        'optuna_direction': "maximize",
        "n_trials": 100
    }
}

MLP_INFORMATION = {
    'model_name': 'Multilayer Perceptron',
    'potential_hyperparameters': {
        'hidden_layer_sizes': {
            'finetune': True,
            'low_value': 50,
            'high_value': 200, #100 is default #128 + 1 = 129/2 = 64.5 ish
            'exact_value': None,
            'trial': 'int',
            'use_log': False,
            'finetuning_step': 10
        },
        'activation': {
            'finetune': False,  # Set to False since we are not fine-tuning this parameter
            'exact_value': 'relu',  # Specify 'relu' as the exact value
        },
        'solver': {
            'finetune': False,
            'trial': 'categorical',  # Set to categorical for solver
            'choices': ['adam', 'lbfgs'],
            'exact_value': 'adam'
        },
        'alpha': {
            'finetune': True,
            'low_value': 0.0001,
            'high_value': 0.01,
            'exact_value': None,
            'trial': 'float',
            'use_log': False,
            'finetuning_step': None
        },
        'learning_rate_init': {
            'finetune': True,
            'low_value': 0.0001,
            'high_value': 0.01,
            'exact_value': None,
            'trial': 'float',
            'use_log': False,
            'finetuning_step': None
        },
        'max_iter': {
            'finetune': True,
            'low_value': 200,
            'high_value': 700,
            'exact_value': 200,
            'trial': 'int',
            'use_log': False,
            'finetuning_step': 50
        },
        'random_state': {
            'finetune': False,  # Set to False since we are not fine-tuning this parameter
            'exact_value': 42,  # Specify '42' as the exact value
        },
    },
    'tuning_options': {
        'cv_number': 5,
        'optuna_direction': "maximize",
        "n_trials": 100
    }
}

LGBM_INFORMATION = {
    'model_name': 'LightGBM',
    'potential_hyperparameters': {
        'n_estimators': {
            'finetune': True,
            'low_value': 100,
            'high_value': 500,
            'exact_value': 100, #default is 100
            'trial': 'int',
            'use_log': False,
            'finetuning_step': 50
        },
        'num_leaves': {
            'finetune': True,
            'low_value': 2,
            'high_value': 2**8, 
            'exact_value': 31, # default is 31
            'trial': 'int',
            'use_log': True,
            'finetuning_step': 1
        },
        'learning_rate': {
            'finetune': True,
            'low_value': 0.001,
            'high_value': 0.2,
            'exact_value': 0.1, # Default is 1
            'trial': 'float',
            'use_log': False,
            'finetuning_step': None
        },
        'subsample': {
            'finetune': True,
            'low_value': 0.2,
            'high_value': 1.0,
            'exact_value': 1.0, # Default is 1
            'trial': 'float',
            'use_log': False,
            'finetuning_step': 0.05
        },
        'colsample_bytree': {
            'finetune': True,
            'low_value': 0.05,
            'high_value': 1.0,
            'exact_value': 1.0, # Default is 1
            'trial': 'float',
            'use_log': False,
            'finetuning_step': 0.05
        },
        'min_child_samples': { # Minimum amount of data in a child
            'finetune': True,
            'low_value': 1,
            'high_value': 30,
            'exact_value': 20, # Default is 20
            'trial': 'int',
            'use_log': False,
            'finetuning_step': 1
        },
        'reg_alpha': { # L1 regularisation parameter
            'finetune': True,
            'low_value': 1e-8,
            'high_value': 10,
            'exact_value': 1e-8, # Default is 0
            'trial': 'float',
            'use_log': True,
            'finetuning_step': None
        },
        'reg_lambda': { # L2 regularisation parameter
            'finetune': True,
            'low_value': 1e-8,
            'high_value': 10,
            'exact_value': 1e-8, # Default is 0
            'trial': 'float',
            'use_log': True,
            'finetuning_step': None
        },
        'random_state': {
            'finetune': False,  # Set to False since we are not fine-tuning this parameter
            'exact_value': 42,  # Specify '42' as the exact value
        },
        'verbose':{
            'finetune': False,  # Set to False since we are not fine-tuning this parameter
            'exact_value': -1,  # -1 to have no verbose
        }

    },
    'tuning_options': {
        'cv_number': 5,
        'optuna_direction': "maximize",
        "n_trials": 100
    }
}

LOGISTIC_REGRESSION_INFORMATION = {
    'model_name': 'Logistic Regression',
    'potential_hyperparameters': {
        'penalty': {
            'finetune': True,
            'choices': ['l2', None],
            'exact_value': 'l2',
            'trial': 'categorical'
        },
        'C': {
            'finetune': True,
            'low_value': 0.1,
            'high_value': 10.0,
            'exact_value': 1.0,
            'trial': 'float',
            'use_log': False,
            'finetuning_step': 0.1
        },
        'solver': {
            'finetune': True,
            'choices': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            'exact_value': 'lbfgs',
            'trial': 'categorical'
        },
        'random_state': {
            'finetune': False,
            'exact_value': 42
        }
    },
    'tuning_options': {
        'cv_number': 3,
        'optuna_direction': "maximize",
        'n_trials': 100
    }
}

SVM_INFORMATION = {
  'model_name': 'SVM',
  'potential_hyperparameters': {
    'C': {
      'finetune': True,
      'low_value': 0.01,
      'high_value': 100.0,
      'exact_value': 1.0,
      'trial': 'float',
      'use_log': True,
      'finetuning_step': None
    },
    'kernel': {
      'finetune': True,
      'choices': ['linear', 'poly', 'rbf', 'sigmoid'],
      'exact_value': 'rbf',
      'trial': 'categorical'
    },
    'degree': {
      'finetune': True,  # Only applicable for poly kernel
      'low_value': 2,
      'high_value': 5,
      'exact_value': 3,
      'trial': 'int',
      'use_log': False,
      'finetuning_step': 1
    },
    'coef0': {
      'finetune': True,  # Only applicable for poly and sigmoid kernel
      'low_value': 0.1,
      'high_value': 1.0,
      'exact_value': 0.1,
      'trial': 'float',
      'use_log': True,
      'finetuning_step': None
    },
    'random_state': {
      'finetune': False,
      'exact_value': 42
    }
  },
  'tuning_options': {
    'cv_number': 3,
    'optuna_direction': "maximize",
    'n_trials': 100
  }
}

LIME_KERNEL_WIDTH = 5