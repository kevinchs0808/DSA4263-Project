import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import math


## Plotting functions

def plot_simple_bar_chart(data, column, target_column):
    """
    Plot the count of fraud reported by insured sex.

    Parameters:
    data (DataFrame): Input DataFrame
    column (str): Name of the column for analysis
    target_column (str): Name of the fraud target column

    Output:
    Plot of the count of fraud reported by the specified column.
    """
    # Set seaborn style to whitegrid
    sns.set_style("whitegrid")

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=column, hue=target_column)
    plt.title('Fraud Reported by {}'.format(column))
    plt.xlabel('{}'.format(column))
    plt.ylabel('Count')
    plt.legend(title='Fraud Reported', loc='upper right')

    # Rotate x-axis labels vertically
    plt.xticks(rotation=90)
    plt.show()

def generate_summary_statistics(data, column, target_column):
    """
    Generate summary statistics for the specified column with respect to the target column.

    Parameters:
    data (DataFrame): Input DataFrame.
    column (str): Name of the column for summary statistics.
    target_column (str): Name of the target column.

    Returns:
    DataFrame: Summary statistics DataFrame.
    """
    # Group by the specified column and calculate summary statistics
    summary_stats = data.groupby(target_column)[column].describe().reset_index()

    return summary_stats

def plot_histogram(data, column, hue_column):
    """
    Plot histogram analysis of the specified column with hue differentiation.

    Parameters:
    data (DataFrame): Input DataFrame.
    column (str): Name of the column for histogram analysis.
    hue_column (str): Name of the column to differentiate with hue.
    """
    # Set seaborn style to whitegrid
    sns.set_style("whitegrid")

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=column, hue=hue_column, element="step", bins=30, kde=True)
    plt.title('Histogram Analysis of {}'.format(column))
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend(title=hue_column, loc='upper right')
    plt.show()


def calculate_fraud_proportion(df, column, target_column):
    # Group by the specified column and count occurrences of target_column
    grouped_data = df.groupby(column)[target_column].value_counts().unstack().reset_index()

    # Calculate the total count of fraud reported for each category
    grouped_data['total_count'] = grouped_data[['N', 'Y']].sum(axis=1)

    # Calculate the proportion of 'Y' fraud reported for each category
    grouped_data['fraud_proportion'] = grouped_data['Y'] / grouped_data['total_count']

    # Sort the DataFrame based on the fraud_proportion column
    sorted_data = grouped_data.sort_values(by='fraud_proportion', ascending=False)

    # Reset index
    sorted_data.reset_index(drop=True, inplace=True)

    sorted_data = sorted_data.rename_axis(None, axis = 1)

    return sorted_data

def calculate_fraud_count(df, column, target_column):
    # Group by the specified columns and count occurrences
    grouped_data = df.groupby([column, target_column]).size().reset_index(name='count')

    grouped_data1 = grouped_data[grouped_data[target_column] == 'Y']

    # Sort the grouped DataFrame based on the count of target_column
    sorted_data = grouped_data1.sort_values(by=['count'], ascending=False)

    x_axis_order = list(sorted_data[column].unique())

    # Reset index
    sorted_data.reset_index(drop=True, inplace=True)

    return sorted_data, x_axis_order

def plot_line_bar_chart(df, var_column, target_column, num_highlight = 2, y_tick = 2):
    """
    Plot the count of fraud reported by insured sex.

    Parameters:
    data (DataFrame): Input DataFrame
    var_column (str): Name of the column for analysis
    target_column (str): Name of the fraud target column
    num_highlight (int): Number of highlighted bars in the bar chart

    output:
    Plot of the count of fraud reported by the specified column. This is a bar chart with a line graph.
    """

    fraud_count_df, x_axis_order = calculate_fraud_count(df, var_column, target_column)

    fraud_proportion_df = calculate_fraud_proportion(df, var_column, target_column)

    # Left join test_df1 and test_df2 on the 'insured_hobbies' column
    merged_df = pd.merge(
        fraud_count_df,
        fraud_proportion_df[[var_column, 'fraud_proportion']],
        on=var_column, how='left'
        )

    # Set seaborn style to whitegrid
    sns.set_style("whitegrid")

    # Plotting
    plt.figure(figsize=(10, 6), dpi = 600)
    ax1 = sns.barplot(
        data=merged_df, x=var_column, y='count',
        hue = var_column, order=x_axis_order,
        palette=['#71D1F0' if i < num_highlight else 'grey' for i in range(len(x_axis_order))])
    # Calculate the maximum count value to determine the y-axis ticks
    max_count = merged_df['count'].max()
    # Set y-axis ticks for ax1
    ax1.set_yticks(np.arange(0, max_count+1, y_tick))
    ax1.set_ylabel('Fraud count')
    # Rotate x-axis labels of ax1
    ax1.tick_params(axis='x', labelrotation=90)
    ax2 = ax1.twinx()
    ax2 = sns.lineplot(
        data=merged_df, x=var_column, y='fraud_proportion',
        color='#1F40CA', linestyle='-')
    # Set y-axis range for ax2
    ax2.set_ylim(0, 1)
    # Set y-axis label for the line graph
    ax2.set_ylabel('Fraud proportion')
    # Remove horizontal grid lines for ax2
    ax2.grid(False)

    plt.title('Fraud Reported and Proportion by {}'.format(var_column))
    plt.xlabel('{}'.format(var_column))

    plt.show()

### Summary Stats Helper

def shortlist_columns(df, columns_list):
    """
    Create a new DataFrame containing only the specified columns from the original DataFrame.

    Parameters:
    df (DataFrame): Original DataFrame.
    columns_list (list of str): List of column names to be selected.

    Returns:
    DataFrame: New DataFrame containing only the specified columns.
    """
    df1 = df.copy()  # Make a copy of the original DataFrame to avoid modifying it
    df2 = df1[columns_list]  # Select only the specified columns

    return df2

def add_date_columns(data):
    """
    Add columns for day of the week, week number for the month, and month based on the 'incident_date' column.

    Parameters:
    data (DataFrame): Input DataFrame containing the 'incident_date' column.

    Returns:
    DataFrame: DataFrame with additional columns added.
    """
    data1 = data.copy()
    # Convert 'incident_date' column to datetime type
    data1['incident_date'] = pd.to_datetime(data1['incident_date'])

    # Add column for day of the week (e.g., Monday, Tuesday, etc.)
    data1['day_of_week'] = data1['incident_date'].dt.strftime('%A')

    # Add column for week number of the month
    # Calculate week number of the month using isocalendar().week
    first_day_of_month = data1['incident_date'].dt.to_period('M').dt.start_time
    week_number_of_month = data1['incident_date'].dt.isocalendar().week - first_day_of_month.dt.isocalendar().week + 1
    data1['week_number'] = week_number_of_month

    # Add column for month (as alphabet)
    data1['month'] = data1['incident_date'].dt.strftime('%B')

    return data1


### Summary statistics 

def shortlist_columns(df, columns_list):
    """
    Create a new DataFrame containing only the specified columns from the original DataFrame.

    Parameters:
    df (DataFrame): Original DataFrame.
    columns_list (list of str): List of column names to be selected.

    Returns:
    DataFrame: New DataFrame containing only the specified columns.
    """
    df1 = df.copy()  # Make a copy of the original DataFrame to avoid modifying it
    df2 = df1[columns_list]  # Select only the specified columns

    return df2

###############################################################################

def categorize_age(age):
    """
    Categorizes the given age into specific age groups.

    Parameters:
    age (int): The age to be categorized.

    Returns:
    str: The age group category that the given age falls into.
    """
    if age >= 20 and age <= 29:
        return '20 to 29'
    elif age >= 30 and age <= 39:
        return '30 to 39'
    elif age >= 40 and age <= 49:
        return '40 to 49'
    else:
        return '50 and above'

def create_age_category_col(df, age_column_name, age_category_name):
    """
    Creates a new age category column based on the values in the specified age column.
    The new age category column is inserted beside the original age column in the DataFrame,
    and the original age category column is removed.

    Parameters:
    df (DataFrame): Input DataFrame containing the age column.
    age_column_name (str): Name of the age column in the DataFrame.
    age_category_name (str): Name for the new age category column to be created.

    Returns:
    DataFrame: DataFrame with the new age category column added and the original age category column removed.
    """
    df1 = df.copy()
    # Assuming df is your DataFrame
    df1[age_category_name] = df1[age_column_name].apply(categorize_age)

    # Get the index of the age column
    age_column_index = df1.columns.get_loc(age_column_name)

    # Insert the age category column beside the age column
    df1.insert(age_column_index + 1, age_category_name, df1.pop(age_category_name))

    return df1

###############################################################################

def create_policy_tenure_col(df, incident_date_column, policy_bind_date_column, policy_tenure_name):
    """
    Calculates the tenure of the policy in years based on the difference between 'incident_date' and 'policy_bind_date'.
    Adds a new column for policy tenure to the DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame containing 'incident_date' and 'policy_bind_date' columns.
    incident_date_column (str): Name of the column containing incident dates.
    policy_bind_date_column (str): Name of the column containing policy bind dates.
    policy_tenure_name (str): Name for the new column representing policy tenure.

    Returns:
    DataFrame: DataFrame with the new policy tenure column added.
    """
    df1 = df.copy()

    # Convert 'incident_date' and 'policy_bind_date' columns to datetime
    df1[incident_date_column] = pd.to_datetime(df1[incident_date_column])
    df1[policy_bind_date_column] = pd.to_datetime(df1[policy_bind_date_column])

    # Calculate the difference in days
    df1['days_difference'] = (df1[incident_date_column] - df1[policy_bind_date_column]).dt.days

    # Convert days to years and rename the column
    df1[policy_tenure_name] = (df1['days_difference'] / 365.25).round(2)  # accounting for leap years

    # Drop the 'days_difference' column
    df1.drop(columns=['days_difference'], inplace=True)

    # Get the index position of the incident_date_column
    incident_date_index = df1.columns.get_loc(incident_date_column)

    # Insert the 'Policy tenure' column beside the incident_date_column
    df1.insert(incident_date_index + 1, policy_tenure_name, df1.pop(policy_tenure_name))

    return df1

###############################################################################

def create_incident_year_col(df, date_column, incident_year_column = 'incident_year'):
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
    df_copy[incident_year_column] = pd.to_datetime(df_copy[date_column]).dt.year

    # Get the index position of the incident_date column
    column_index = df_copy.columns.get_loc(date_column)

    # Drop incident_date column
    # df_copy.drop(column, axis=1, inplace=True)

    # Insert the new column at the original index position of the incident_date column
    df_copy.insert(column_index, incident_year_column, df_copy.pop(incident_year_column))

    return df_copy

###############################################################################

def create_car_age_col(df, incident_year_column, auto_year_column, car_age_name):
    """
    Calculates the age of the car at the time of the incident based on the incident year and the car's manufacturing year.
    Adds a new column for car age to the DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame.
    incident_year_column (str): Name of the column containing the incident year.
    auto_year_column (str): Name of the column containing the car's manufacturing year.
    car_age_name (str): Name for the new column representing car age.

    Returns:
    DataFrame: DataFrame with the new car age column added.
    """
    df1 = df.copy()
    df1[car_age_name] = df1[incident_year_column] - df1[auto_year_column]
    # Get the index position of the incident_date column
    column_index = df1.columns.get_loc(auto_year_column)
    # Insert the new column at the original index position of the incident_date column
    df1.insert(column_index, car_age_name, df1.pop(car_age_name))
    return df1

###############################################################################

def drop_cols(df, columns_list):
    """
    Drops the specified columns from the DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame.
    columns_list (list): List of column names to be dropped from the DataFrame.

    Returns:
    DataFrame: DataFrame with specified columns dropped.
    """
    df1 = df.copy()
    df1.drop(columns=columns_list, inplace=True)
    return df1

def create_categorical_summary_statistics_df(df, columns_key_list, column):
    """
    Creates a summary DataFrame by grouping the input DataFrame based on specified key columns
    and counting the occurrences of values in the specified column.

    Parameters:
    df (DataFrame): Input DataFrame.
    columns_key_list (list): List of column names to group by.
    column (str): Name of the categorical column whose values will be counted.

    Returns:
    DataFrame: Summary DataFrame containing counts of values in the specified column
               grouped by key columns.
    """
    df1 = df.groupby(columns_key_list)[column].value_counts().reset_index()
    return df1

################################################################################
# Define a function to calculate confidence intervals
def confidence_interval(data):
    """
    Calculate the 95% confidence interval for a given data set.

    Parameters:
    data (list): A list of numerical data.

    Returns:
    tuple: A tuple containing the lower and upper bounds of the 95% confidence interval, rounded to 2 decimal places.
    """
    mean = np.mean(data)
    std_err = stats.sem(data)
    z_score = stats.norm.ppf(0.975)  # 95% confidence level, two-tailed
    interval = (z_score * std_err) / math.sqrt(len(data))
    return round(mean - interval, 2), round(mean + interval, 2)

def create_numerical_summary_statistics_df(df, age_interval_column, fraud_column, columns_list):
    """
    Create a dataframe with summary statistics for each combination of age interval and fraud status.

    Parameters:
    df (DataFrame): The input dataframe.
    age_interval_column (str): The name of the column in df representing the age interval.
    fraud_column (str): The name of the column in df representing the fraud status.
    columns_list (list): A list of column names for which to calculate the confidence intervals.

    Returns:
    DataFrame: A dataframe with one row for each combination of age interval and fraud status, and one column for the 95% confidence interval of each column in columns_list.
    """
    # Obtain the unique age_interval list
    unique_age_interval_list = list(df[age_interval_column].unique())
    # Obtain the unique fraud list
    unique_fraud_column_list = list(df[fraud_column].unique())

    # Obtain the grouped column
    grouped_df = df.groupby([age_interval_column, fraud_column]).size().reset_index(name='count')

    for column in columns_list:
        confint_list = []
        for fraud in unique_fraud_column_list:
            for age_interval_entry in unique_age_interval_list:
                filtered_df = df[(df[age_interval_column] == age_interval_entry) &\
                                (df[fraud_column] == fraud)]
                confidence_interval_tuple = confidence_interval(list(filtered_df[column]))
                confint_list.append(confidence_interval_tuple)

        temp_column = '95% Confidence Interval' + ' ' + column
        grouped_df[temp_column] = confint_list

    # Drop the 'count' column
    grouped_df.drop(columns=['count'], inplace=True)
    return grouped_df


################################################################################

def shortlist_most_popular_category_df(df, has_fraud):
    """
    Filters the DataFrame based on whether fraud is reported or not ('Y' or 'N')
    and shortlists the top 3 values for each age interval.

    Parameters:
    df (DataFrame): Input DataFrame.
    has_fraud (str): Specifies whether fraud is reported ('Y') or not ('N').

    Returns:
    DataFrame: DataFrame containing the top 3 values for each age interval
               within the filtered DataFrame based on fraud report.
    """
    # Filter the DataFrame based on whether fraud is reported or not
    filtered_df = df[df['fraud_reported'] == has_fraud].copy()

    # Get the top 3 values for each age interval
    top_3_values = filtered_df.groupby('age interval').apply(lambda x: x.nlargest(3, 'count')).reset_index(drop=True)

    return top_3_values