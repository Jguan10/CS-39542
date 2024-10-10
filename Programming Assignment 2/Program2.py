"""
Pandas version 2.2.3
Numpy version 2.1.2
Matplotlib version 3.9.2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def import_data(file_name):
    """
    Read relevant data from DOE High School Directory CSV file.

    This function should only select the following columns, preserving their order:
    [
            "dbn",
            "school_name",
            "borocode",
            "NTA",
            "graduation_rate",
            "pct_stu_safe",
            "attendance_rate",
            "college_career_rate",
            "language_classes",
            "advancedplacement_courses"
    ]

    Any rows with missing values for `graduation rate` should be dropped 
    from the returned dataframe.

    :param file_name: name/path to DOE High School Directory CSV file.
    :type file_name: str.
    :returns: relevant data from specified CSV file.
    :rtype: pandas.DataFrame.
    """

    df = pd.read_csv(file_name)
    filtered_df = df[
        [
            "dbn",
            "school_name",
            "borocode",
            "NTA",
            "graduation_rate",
            "pct_stu_safe", 
            "attendance_rate",
            "college_career_rate",
            "language_classes",
            "advancedplacement_courses",
        ]
    ]
    filtered_df = filtered_df.dropna(subset=["graduation_rate"])

    return filtered_df


def impute_numeric_cols_median(df):
    """
    Impute missing numeric values with column median.
    Any missing entries in the numeric columns 
    ["pct_stu_safe","attendance_rate", "college_career_rate"]
    are replaced with the median of the respective column"s non-missing values.

    :param df: dataframe ontaining DOE High School from OpenData NYC.
    :type df: pandas.DataFrame.
    :returns: dataframe with imputed values.
    :rtype: pandas.DataFrame.
    """

    df["pct_stu_safe"] = df["pct_stu_safe"].fillna(df["pct_stu_safe"].median())
    df["attendance_rate"] = df["attendance_rate"].fillna(df["attendance_rate"].median())
    df["college_career_rate"] = df["college_career_rate"].fillna(
        df["college_career_rate"].median()
    )

    return df


def compute_item_count(df, col):
    """
    Counts the number of items, separated by commas, in each entry of `df[col]`.

    :param df: dataframe containing DOE High School from OpenData NYC.
    :type df: pandas.DataFrame.
    :col: a column key in `df` that contains a list of items separated by
            commas.
    :type col: str.
    :returns series with
    :rtype: pandas.Series.

    Example:
    >>> pets_df = pd.DataFrame({"name": ["Abdullah", "Betty", "Carmen"],  
    "animals":  ["cat", "canary, dog", "dog, dog, dog"]})
    >>> compute_item_count(pets_df, "animals")
    0    1
    1    2
    2    3
    Name: animals, dtype: int64
    """

    computed = df[col].apply(lambda x: len(x.split(",")) if isinstance(x, str) else 0)

    return computed


def encode_categorical_col(x):
    """
    One-hot encode a categorical column.

    Takes a column of categorical data and performs one-hot encoding to create a
    new DataFrame with k columns, where k is the number of distinct values in the
    column. Output columns should have the same ordering as their values would if
    sorted.

    :param x: series containing categorical data.
    :type x: pandas.Series.
    :returns: dataframe of categorical encodings of x.
    :rtype: pandas.DataFrame


    NOTE: in the lectures we presented several different ways to encode categorical
          data. The DS 100 book details an approach in Chapter 15 based on the
          scikit-learn (sklearn) library. We will use scikit-learn in future
          programs, but in this one you should use only pandas and the python standard
          library. You may find pandas" `get_dummies()` function useful:
              https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html


    Example:
    >>> names = pd.Series(["Carmen", "Abdullah", "Betty"])
    >>> encode_categorical_col(names)
       Abdullah  Betty  Carmen
    0         0      0       1
    1         1      0       0
    2         0      1       0
    """

    df = pd.Series(x)
    encoded = df.str.get_dummies(sep=", ")
    return encoded


def split_test_train(df, x_col_names, y_col_name, frac=0.25, random_state=922):
    """
    Split data into train and test subsets.

    Suggestions for algorithm:
            1. Create a new DataFrame, `test_df` using the DataFrame"s `.sample()` method to
               create a test set with `frac` of the rows using `random_state` as the random
               number generator seed.
            2. Make a copy of the DataFrame, `df` and call it `train_df`.
            3. Drop/remove the rows in `test_df` from `train_df`.
            4. Return the tuple 
            `(train_df[x_col_names], test_df[x_col_names], 
            train_df[y_col_name], test_df[y_col_name])`

    NOTE: we are using only pandas for this assignment. In future assignments, we will use
              the scikit-learn packages to split data into training and testing sets.

    :param df: dataframe containing input columns (aka independent variables,
            predictors, features, covariates, ...) and output column (aka dependent
            variable, target, ...).
    :type df: pandas.DataFrame.
    :param x_col_names: column keys to input variables.
    :type x_col_names: list or iterable.
    :param y_col_name: column key to output variable.
    :type y_col_name: str.
    :param frac: fraction (between 0 and 1) of the data for the test set. Defaults
            to 0.25.
    :type frac: float.
    :param random_state: random generator seed. Defaults to 922.
    :type random_state: int.
    :returns: a tuple (x_train, x_test, y_train, y_test) with selected columns of
            the original data in df split into train and test sets.
    :rtype: tuple(pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series)

    Example:
    >>> df = pd.DataFrame({"x1": [0, 1, 2, 3],  "x2": [0, 1, -1, 3], "y": [-1, 0, 4, 10]})
    split_test_train(df, ["x1", "x2"], "y")
    (   x1  x2
     0   0   0
     2   2  -1
     3   3   3,
            x1  x2
     1   1   1,
     0    -1
     2     4
     3    10
     Name: y, dtype: int64,
     1    0
     Name: y, dtype: int64)
    """

    df_test = df.sample(frac=frac, random_state=random_state)
    df_train = df.copy()
    df_train = df_train.drop(df_test.index)

    x_train = df_train[x_col_names]
    x_test = df_test[x_col_names]
    y_train = df_train[y_col_name]
    y_test = df_test[y_col_name]

    return x_train, x_test, y_train, y_test


def compute_lin_reg(x, y):
    """
    :param x: 1-dimensional array containing the predictor (independent) variable values.
    :type x: pandas.Series, numpy.ndarray, or iterable of numeric values.
    :param y: 1-dimensional array containing the predictor (independent) variable values.
    :type y: pandas.Series, numpy.ndarray, or iterable of numeric values.
    :return: tuple containing the model"s (intercept, slope)
    :rtype: tuple(float, float)
    The function computes the slope and 
    y-intercept of the 1-d linear regression line,
    using ordinary least squares (see DS 8, Chapter 15 or DS 100, 
    Chapter 15 for detailed explanation.

    Algorithm for this:
            1. Compute the standard deviation of `x` and `y`. Call these `sd_x` and `sd_y`.
            2. Compute the correlation, `r`, between `x` and `y`.
            3. Compute the slope, `theta_1`, as `theta_1 = r*sd_y/sd_x`.
            4. Compute the intercept, `theta_0`, as
           `theta_0 = average(yes) - theta_1 * average(x)`* Return `theta_0` and `theta_1`.
    """

    std_x = x.std()
    std_y = y.std()
    r = x.corr(y)
    theta_1 = r * (std_y / std_x)
    theta_0 = y.mean() - theta_1 * x.mean()

    return theta_0, theta_1


def predict(x, theta_0, theta_1):
    """
        Make 1-d linear model prediction on an array of inputs.
    %
     The function returns the predicted values of the dependent variable, `x`, under
     the linear regression model with y-intercept `theta_0` and slope `theta_1`

    :param x: array of numeric values representing the independent variable.
    :type x: pandas.Series or numpy.ndarray.
    :param theta_0: the y-intercept of the linear regression model.
    :type theta_0: float
    :param theta_1: the slope of the 1-d linear regression model.
    :type theta_1: float
    :returns: array of numeric values with the predictions y = theta_0 + theta_1 * x.
    :rtype: pandas.Series or numpy.ndarray.
    """

    return x.apply(lambda x: theta_0 + theta_1 * x)


def mse_loss(y_actual, y_estimate):
    """
        Compute the MSE loss.

    :param y_actual: numeric values representing the actual observations of
        the dependent variable.
    :type y_actual: pandas.Series or numpy.ndarray.
    :param y_estimate: numeric values representing the model predictions for
        the dependent variable.
    :type y_estimate: pandas.Series or numpy.ndarray.
    :returns: MSE loss between y_actual and y_estimate.
    :rtype: float.
    """

    return np.mean((y_estimate - y_actual) ** 2)


def rmse_loss(y_actual, y_estimate):
    """
        Compute the RMSE loss.

    :param y_actual: numeric values representing the actual observations of
        the dependent variable.
    :type y_actual: pandas.Series or numpy.ndarray.
    :param y_estimate: numeric values representing the model predictions for
        the dependent variable.
    :type y_estimate: pandas.Series or numpy.ndarray.
    :returns: RMSE loss between y_actual and y_estimate.
    :rtype: float.
    """

    return np.mean((y_estimate - y_actual) ** 2) ** 0.5


def compute_loss(y_actual, y_estimate, loss_fnc=mse_loss):
    """
        Compute a user-specified loss.

    :param y_actual: numeric values representing the actual observations of
        the dependent variable.
    :type y_actual: pandas.Series or numpy.ndarray.
    :param y_estimate: numeric values representing the model predictions for
        the dependent variable.
    :param loss_fnc: a loss function. Defaults to `mse_loss`.
    :type loss_fnc: function.
    :type y_estimate: pandas.Series or numpy.ndarray.
    :returns: RMSE loss between y_actual and y_estimate.
    :rtype: float.
    """

    if loss_fnc == "rmse_loss":
        return rmse_loss(y_actual, y_estimate)
    return mse_loss(y_actual, y_estimate)

file_name = "2021_DOE_High_School_Directory_SI.csv"
si_df = import_data(file_name)
print(f"There are {len(si_df.columns)} columns:")
print(si_df.columns)
print("The dataframe is:")
print(si_df)

file_name = "2020_DOE_High_School_Directory_late_start.csv"
late_df = import_data(file_name)
print("The numerical columns are:")
print(late_df[["dbn", "pct_stu_safe", "attendance_rate", "college_career_rate"]])

late_df = impute_numeric_cols_median(late_df)
print(late_df[["dbn", "pct_stu_safe", "attendance_rate", "college_career_rate"]])

late_df["language_count"] = compute_item_count(late_df, "language_classes")
late_df["ap_count"] = compute_item_count(late_df, "advancedplacement_courses")
print("High schools that have 9am or later start:")
print(
    late_df[
        [
            "dbn",
            "language_count",
            "language_classes",
            "ap_count",
            "advancedplacement_courses",
        ]
    ]
)

boros_df = encode_categorical_col(late_df["borocode"])
print(late_df["borocode"].head(5))
print(boros_df.head(5))
print("Number of schools in each borough:")
print(boros_df.sum(axis=0))

x_cols = [
    "language_count",
    "ap_count",
    "pct_stu_safe",
    "attendance_rate",
    "college_career_rate",
]
y_col = "graduation_rate"
x_train, x_test, y_train, y_test = split_test_train(late_df, x_cols, y_col)
print(f"The sizes of the sets are:")
print(f"x_train has {len(x_train)} rows.\tx_test has {len(x_test)} rows.")
print(f"y_train has {len(y_train)} rows.\ty_test has {len(y_test)} rows.")

x_cols = [
    "language_count",
    "ap_count",
    "pct_stu_safe",
    "attendance_rate",
    "college_career_rate",
]
coeff = {}
for col in x_cols:
    coeff[col] = compute_lin_reg(x_train[col], y_train)
    print(f"For {col}, theta_0 = {coeff[col][0]} and theta_1 = {coeff[col][1]}")

# loop through all the models, computing train and test loss
y_train_predictions = {}
y_test_predictions = {}
train_losses = {}
test_losses = {}
min_loss = 1e09
for col in x_cols:
    theta_0, theta_1 = coeff[col]
    y_train_predictions[col] = predict(x_train[col], theta_0, theta_1)
    y_test_predictions[col] = predict(x_test[col], theta_0, theta_1)

    train_losses[col] = compute_loss(y_train, y_train_predictions[col])
    test_losses[col] = compute_loss(y_test, y_test_predictions[col])

# arrange models" train and test losses into a dataframe
losses_df = pd.DataFrame(
    index=x_cols,
    data={
        "train_loss": train_losses.values(),
        "test_loss": test_losses.values(),
    },
)
print(losses_df)


def graph_data(df, col, coeff):
    """
    Function to graph the models
    """
    plt.scatter(df[col], df["graduation_rate"], label="Actual")
    predict_grad = predict(df[col], coeff[col][0], coeff[col][1])
    plt.scatter(df[col], predict_grad, label="Predicted")
    plt.title(f"{col} vs graduation_rate")
    plt.ylabel("graduation_rate")
    plt.xlabel(f"{col}")
    plt.legend()
    plt.show()


graph_data(late_df, "college_career_rate", coeff)
