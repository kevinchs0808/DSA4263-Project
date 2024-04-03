import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTENC, RandomOverSampler, ADASYN
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def one_hot_encode(df, column):
    """
    Perform one-hot encoding on a specified column of a DataFrame
    while keeping the original column index position intact.

    Parameters:
    df (DataFrame): Input DataFrame.
    column (str): Name of the column to one-hot encode.

    Returns:
    DataFrame: DataFrame with the specified column one-hot encoded and
    original column removed.
    """
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Perform one-hot encoding
    encoded_df = pd.get_dummies(df_copy[column], prefix=column)

    # Get the original column index position
    column_index = df_copy.columns.get_loc(column)

    # Drop the original column from the DataFrame
    df_copy = df_copy.drop(column, axis=1)

    # Insert the one-hot encoded columns at the original column index position
    for col in encoded_df.columns:
        df_copy.insert(column_index, col, encoded_df[col])

    return df_copy

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

def perform_StandardScaling(df, columns_list):
    """
    Performs Standard Scaling on specified columns of the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    columns_list (list): List of column names to be standardized.

    Returns:
    DataFrame: The DataFrame with specified columns standardized.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns_list] = scaler.fit_transform(df_scaled[columns_list])
    return df_scaled


def perform_MinMaxScaling(df, columns_list):
    """
    Performs Min-Max Scaling on specified columns of the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    columns_list (list): List of column names to be scaled.

    Returns:
    DataFrame: The DataFrame with specified columns scaled.
    """
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[columns_list] = scaler.fit_transform(df_scaled[columns_list])
    return df_scaled


def perform_RobustScaling(df, columns_list):
    """
    Performs Robust Scaling on specified columns of the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    columns_list (list): List of column names to be scaled.

    Returns:
    DataFrame: The DataFrame with specified columns scaled.
    """
    scaler = RobustScaler()
    df_scaled = df.copy()
    df_scaled[columns_list] = scaler.fit_transform(df_scaled[columns_list])
    return df_scaled

def perform_scaling(df, columns_list, scaling_type):
    """
    Performs scaling on specified columns of the DataFrame based on the specified scaling type.

    Parameters:
    df (DataFrame): The input DataFrame.
    columns_list (list): List of column names to be scaled.
    scaling_type (str): Type of scaling to be performed. Options: 'standard', 'minmax', 'robust'.

    Returns:
    DataFrame: The DataFrame with specified columns scaled.
    """
    if scaling_type == 'standard':
        return perform_StandardScaling(df, columns_list)
    elif scaling_type == 'minmax':
        return perform_MinMaxScaling(df, columns_list)
    elif scaling_type == 'robust':
        return perform_RobustScaling(df, columns_list)
    else:
        raise ValueError("Invalid scaling_type. Please choose from 'standard', 'minmax', or 'robust'.")

################################################################################

def process_incident_date(df, column):
    """
    Extracts the year from the specified incident date column and creates a new column for the year.
    Then drops the incident date column from the DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame containing the incident date column.
    column (str): Name of the incident date column. Default is 'incident_date'.

    Returns:
    DataFrame: DataFrame with the incident date column dropped and a new column
    for the year added.
    """
    df_copy = df.copy()

    # Extract year from incident_date and create a new column
    df_copy['incident_year'] = pd.to_datetime(df_copy[column]).dt.year

    # Get the index position of the incident_date column
    column_index = df_copy.columns.get_loc(column)

    # Drop incident_date column
    # df_copy.drop(column, axis=1, inplace=True)

    # Insert the new column at the original index position of the incident_date column
    df_copy.insert(column_index, 'incident_year', df_copy.pop('incident_year'))

    return df_copy

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

def preprocessing_guan_yee(df, encoding, normalization):
    """
    Preprocesses the input DataFrame by performing one-hot encoding on categorical columns
    and robust scaling on numerical columns if specified.

    Parameters:
    df (DataFrame): The input DataFrame.
    encoding (bool): Flag indicating whether to perform one-hot encoding of categorical features.
    normalization (bool): Flag indicating whether to perform numerical scaling.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    df_processed = df.copy()
     # categorical_columns
    categorical_columns_list = [
        'insured_sex', 'insured_education_level',
        'insured_occupation','insured_hobbies',
        'insured_relationship', 'incident_type',
        'collision_type',
        ]
    numerical_columns_list = [
        'capital-gains', 'capital-loss'
        ]

    if encoding:
        for column in categorical_columns_list:
            df_processed = one_hot_encode(df_processed, column)

    if normalization:
        df_processed = perform_scaling(df_processed,
                                       numerical_columns_list, "robust")
    return df_processed

################################################################################

def preprocessing_kevin(df, encoding, normalization):
    """
    Preprocesses the DataFrame according to the specified encoding and normalization options.

    Parameters:
    df (DataFrame): The input DataFrame.
    encoding (bool): Whether to perform one-hot encoding for the 'policy_state' column.
    normalization (bool): Whether to perform normalization.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    df_processed = df.copy()
    # Categorical Variable Handling
    if encoding == True:
        df_processed = one_hot_encode(df_processed, column = 'policy_state')

    # Convert date into pandas.DateTimeIndex
    df_processed['policy_bind_date'] = pd.to_datetime(df_processed['policy_bind_date'])
    df_processed['incident_date'] = pd.to_datetime(df_processed['incident_date'])

    # Compute the policy age during the incident event
    df_processed['policy_age_during_incident_in_days'] = (df_processed['incident_date'] - df_processed['policy_bind_date']) / np.timedelta64(1, 'D')

    # Obtain numerical values which are represented as string in policy_csl
    df_processed[['bodily_injured_maximum_coverage_per_accident', 'complete_maximum_coverage_per_accident']] = df_processed['policy_csl'].str.split('/', n=1, expand=True)

    # Convert data type from string to integer
    df_processed['bodily_injured_maximum_coverage_per_accident'] = df_processed['bodily_injured_maximum_coverage_per_accident'].astype('int64')
    df_processed['complete_maximum_coverage_per_accident'] = df_processed['complete_maximum_coverage_per_accident'].astype('int64')

    if normalization == True:
        columns_with_standardization = ['months_as_customer', 'age',
                                        'policy_annual_premium']
        columns_with_minmax = ['policy_deductable', 'umbrella_limit',
                               'policy_age_during_incident_in_days',
                               'bodily_injured_maximum_coverage_per_accident',
                               'complete_maximum_coverage_per_accident']

        df_processed = perform_scaling(df_processed,
                                       columns_with_standardization, "standard")
        df_processed = perform_scaling(df_processed,
                                       columns_with_minmax, "minmax")
    return df_processed

################################################################################

def preprocessing_peizhi(df, encoding, normalization):
    """
    Preprocesses the DataFrame according to the specified encoding and normalization options.

    Parameters:
    df (DataFrame): The input DataFrame.
    encoding (bool): Whether to perform one-hot encoding.
    normalization (bool): Whether to perform normalization.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    df_processed = df.copy()
    vars_drop = ['incident_city', 'incident_location']
    vars_onehot_encode = [
        'incident_severity',
        'authorities_contacted',
        'incident_state',
        'property_damage',
        ]

    vars_numerical = [
        'number_of_vehicles_involved',
        'bodily_injuries',
        'incident_hour_of_the_day',
        'witnesses'
        ]

    df_processed = df_processed.drop(vars_drop, axis=1)

    if encoding:
        for var in vars_onehot_encode:
            df_processed = one_hot_encode(df_processed, var)

    if normalization:
        # none of the allocated numberical features are normally distributed, min-max scaling is used
        # No outliers observed also
        df_processed = perform_scaling(df_processed,
                                       vars_numerical, "minmax")

    return df_processed

################################################################################

def create_auto_region_column(df):
    """
    Creates a new column 'auto_region' in the DataFrame indicating the country of origin for each car based on the auto_make column.

    Parameters:
    df (DataFrame): The input DataFrame containing the auto_make column.

    Returns:
    DataFrame: A copy of the input DataFrame with the 'auto_region' column added based on the mapping.
    """
    df_processed = df.copy()
    ## New column Auto Region for which country the car comes from

    country_map = {
        'Acura': 'Japan',
        'Audi': 'Germany',
        'BMW': 'Germany',
        'Chevrolet': 'US',
        'Dodge': 'US',
        'Ford': 'US',
        'Honda': 'Japan',
        'Jeep': 'US',
        'Mercedes': 'Germany',
        'Nissan': 'Japan',
        'Saab': 'Sweden',
        'Suburu': 'Japan', #Typo
        'Toyota': 'Japan',
        'Volkswagen': 'Germany' #Typo
    }

    # Add a new column 'auto_region' based on the mapping
    df_processed['auto_region'] = df_processed['auto_make'].apply(lambda x: country_map.get(x))

    return df_processed

def create_auto_type_column(df):
    """
    Creates a new column 'auto_type' in the DataFrame indicating the type of car based on the auto_model column.

    Parameters:
    df (DataFrame): The input DataFrame containing the auto_model column.

    Returns:
    DataFrame: A copy of the input DataFrame with the 'auto_type' column added based on the mapping.
    """
    ## New column Auto Type for what type the car is (SUV etc.)
    df_processed = df.copy()
    type_map = {
        '92x': 'Hatchback',
        'E400': 'Coupe',
        'RAM': 'Truck',
        'Tahoe': 'SUV',
        'RSX': 'Coupe',
        '95': 'Sedan',
        'Pathfinder': 'SUV',
        'A5': 'Coupe',
        'Camry': 'Sedan',
        'F150': 'Truck',
        'A3': 'Hatchback',
        'Highlander': 'SUV',
        'Neon': 'Hatchback',
        'MDX': 'SUV',
        'Maxima': 'Sedan',
        'Legacy': 'Sedan',
        'TL': 'Sedan',
        'Impreza': 'Sedan',
        'Forrestor': 'SUV',
        'Escape': 'SUV',
        'Corolla': 'Sedan',
        '3 Series': 'Sedan',
        'C300': 'Sedan',
        'Wrangler': 'SUV',
        'M5': 'Sedan',
        'X5': 'SUV',
        'Civic': 'Sedan',
        'Passat': 'Sedan',
        'Silverado': 'Truck',
        'CRV': 'SUV',
        '93': 'Hatchback',
        'Accord': 'Sedan',
        'X6': 'Coupe',
        'Malibu': 'Sedan',
        'Fusion': 'Sedan',
        'Jetta': 'Sedan',
        'ML350': 'SUV',
        'Ultima': 'Sedan',
        'Grand Cherokee': 'SUV'
        }

    # Add a new column 'auto_type' based on the mapping
    df_processed['auto_type'] = df_processed['auto_model'].apply(lambda x: type_map.get(x))

    return df_processed

def preprocessing_vivek(df, encoding, normalization):
    """
    Preprocesses the DataFrame according to the specified encoding and normalization options.

    Parameters:
    df (DataFrame): The input DataFrame.
    encoding (bool): Whether to perform one-hot encoding for categorical variables.
    normalization (bool): Whether to perform normalization.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    df_processed = df.copy()

    ## Additional Features
    df_processed = process_incident_date(df_processed, 'incident_date')
    df_processed['car_age'] = df_processed['incident_year'] - df_processed['auto_year']

    df_processed = create_auto_region_column(df_processed)
    df_processed = create_auto_type_column(df_processed)

    ## auto year and auto model are removed
    vars_cat = [
        'police_report_available',
        'auto_make',
        'auto_region',
        'auto_type'
    ]

    if encoding == True:
        for var in vars_cat:
            df_processed = one_hot_encode(df_processed, var)

    if normalization == True:
        columns_with_robustness = ['total_claim_amount', 'injury_claim',
                                   'property_claim', 'vehicle_claim']
        columns_with_minmax = ['car_age']
        df_processed = perform_scaling(df_processed,
                                       columns_with_robustness, "robust")
        df_processed = perform_scaling(df_processed,
                                       columns_with_minmax, "minmax")
    return df_processed

################################################################################

def preprocessing_drop(df, cols_to_drop_list):
    """
    Drop specified columns from the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    cols_to_drop_list (list): List of column names to be dropped.

    Returns:
    DataFrame: The DataFrame with specified columns dropped.
    """
    df_processed = df.copy()
    df_processed = df_processed.drop(cols_to_drop_list, axis = 1)
    return df_processed

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

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

    Returns:
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
################################################################################
################################################################################
################################################################################
################################################################################

def preprocess_pipeline(df, encoding, normalization):
    """
    Preprocesses the input DataFrame for machine learning modeling.

    Parameters:
    df (DataFrame): The input DataFrame containing features and target variable.
    encoding (bool): Flag indicating whether to perform one-hot encoding of categorical features.
    normalization (bool): Flag indicating whether to perform numerical standardization or normalization.

    Returns:
    DataFrame, DataFrame, Series, Series: Preprocessed training and testing features,
        and corresponding target variables.
    """
    cols_to_drop_list = ["_c39",'policy_number', 'policy_bind_date',
                        'incident_date', 'policy_csl', 'insured_zip',
                         'auto_model', 'auto_year']
    
    df_features = df.drop(['fraud_reported'], axis=1)
    df_label = df['fraud_reported']

    df_label = df_label.replace({'Y': 1, 'N': 0})

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_label,
                                                        stratify = df_label, # Imbalanced dataset, please stratify
                                                        test_size=0.2,
                                                        random_state=42)

    # Numerical Standardization / Normalization (if needed)
    X_train = preprocessing_guan_yee(X_train, encoding, normalization)
    X_train = preprocessing_kevin(X_train, encoding, normalization)
    X_train = preprocessing_peizhi(X_train, encoding, normalization)
    X_train = preprocessing_vivek(X_train, encoding, normalization)
    X_train = preprocessing_drop(X_train, cols_to_drop_list)

    X_test = preprocessing_guan_yee(X_test, encoding, normalization)
    X_test = preprocessing_kevin(X_test, encoding, normalization)
    X_test = preprocessing_peizhi(X_test, encoding, normalization)
    X_test = preprocessing_vivek(X_test, encoding, normalization)
    X_test = preprocessing_drop(X_test, cols_to_drop_list)

    return X_train, X_test, y_train, y_test

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

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
        X_train1, y_train1 = imbalanced_dataset_treatment(X_train, y_train,
                                                          oversampling_strategy)

        # Fit the model to the X_train dataset
        model.fit(X_train1, y_train1)

        # Obtain the prediction for the
        y_pred = model.predict(X_test)

        # Compute the evaluation metric using the scoring function
        score_function = get_scoring_function(metric)
        if metric in ['f1-score', 'recall', 'precision']:
            score = score_function(y_test, y_pred, pos_label = 'Y')
        else:
            score = score_function(y_test, y_pred)
        scores.append(score)

    cv_score = sum(scores) / len(scores)
    return cv_score