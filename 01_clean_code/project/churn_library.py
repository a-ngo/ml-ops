"""
Main script to run the library to predict clients who are likely to churn.
author: a-ngo
date: 2024-05-08
"""

# import libraries
# from sklearn.metrics import plot_roc_curve, classification_report #
# plot_roc_curve is depcrecated
import os
import argparse

from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig('images/churn_histogram.png')

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig('images/customer_age_histogram.png')

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/marital_status.png')

    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/total_trans_ct.png')

    # Need to filter only qunatitative columns
    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]

    plt.figure(figsize=(20, 10))
    sns.heatmap(df[quant_columns].corr(), annot=False,
                cmap='Dark2_r', linewidths=2)
    plt.savefig('images/heatmap.png')


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        # bc this failed: groups = df.groupby(category).mean()['Churn']
        groups = df.groupby(category)["Churn"].mean()

        lst = []
        for val in df[category]:
            lst.append(groups.loc[val])

        df[f'{category}_Churn'] = lst

    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df['Churn']

    X = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]

    # This cell may take up to 15-20 minutes to run
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def save_classification_report(
        model_name,
        y_train,
        y_test,
        y_train_preds,
        y_test_preds):
    """
    Save classification train and test results as an image
    input:
        model_name: name of the model
        y_train: y training data
        y_test: y testing data
        y_train_preds: y training predictions
        y_test_preds: y testing predictions
    output:
        None
    """
    plt.figure(figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str(f'{model_name} Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(f'{model_name} Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(f"images/{model_name}_classification_report.png",
                bbox_inches='tight', pad_inches=0.1, dpi=300)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    output:
        None
    '''
    save_classification_report(
        "Random_Forest", y_train, y_test, y_train_preds_rf, y_test_preds_rf)
    save_classification_report(
        "Logistic_Regression",
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr)


def feature_importance_plot(model, X_data):
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
    output:
        None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(f"images/random_forest_feature_importances.png",
                bbox_inches='tight', pad_inches=0.1, dpi=300)


def plot_roc_curves(X_test, y_test, lr_model, rfc_model):
    """
    plot roc curves for logistic regression and random forest models
    input:
        X_test: X test data
        y_test: y test data
        lr_model: logistic regression model
        rfc_model: random forest model
    output:
        None
    """
    lrc_plot = RocCurveDisplay.from_estimator(lr_model, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(
        rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/roc_curves.png",
                bbox_inches='tight', pad_inches=0.1, dpi=300)


def plot_tree_explainer(rfc_model, X_test):
    """
    plot tree explainer for random forest model
    input:
        rfc_model: random forest model
        X_test: X test data
    output:
        None
    """
    plt.figure(figsize=(20, 10))
    explainer = shap.TreeExplainer(rfc_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
#     shap.force_plot(shap_values, X_test,
#                     matplotlib=True, show=False)
    plt.savefig("images/random_forest_tree_explainer.png",
                bbox_inches='tight', pad_inches=0.1, dpi=300)


def train_or_load_models(X_train, X_test, y_train, y_test, train=False):
    '''
    train, store model results: images + scores, and store models
    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''
    if train:
        print("Starting model training...")
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference:
        # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        lr_model = LogisticRegression(solver='lbfgs', max_iter=3000)

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        rfc = RandomForestClassifier(random_state=42)
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

        cv_rfc.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)

        # Save best models
        rfc_model = cv_rfc.best_estimator_
        joblib.dump(rfc_model, './models/rfc_model.pkl')
        joblib.dump(lr_model, './models/logistic_model.pkl')
    else:
        print("Starting loading saved models...")
        # Load saved models
        rfc_model = joblib.load('./models/rfc_model.pkl')
        lr_model = joblib.load('./models/logistic_model.pkl')

    y_train_preds_rf = rfc_model.predict(X_train)
    y_test_preds_rf = rfc_model.predict(X_test)

    y_train_preds_lr = lr_model.predict(X_train)
    y_test_preds_lr = lr_model.predict(X_test)

    # reports
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    plot_roc_curves(X_test, y_test, lr_model, rfc_model)

    plot_tree_explainer(rfc_model, X_test)

    feature_importance_plot(rfc_model, X_test)


if __name__ == "__main__":
    # Create argument parser object
    parser = argparse.ArgumentParser(
        description="Main script to run the library to predict clients who are likely to churn.")
    parser.add_argument("-t", "--train", action="store_true", default=False,
                        help="Flag to train the models")
    args = parser.parse_args()

    data_frame = import_data(r"./data/bank_data.csv")

    perform_eda(data_frame)

    columns_to_encode = ["Gender", "Education_Level",
                         "Income_Category", "Marital_Status", "Card_Category"]
    data_frame = encoder_helper(data_frame, columns_to_encode)

    X_train, X_test, y_train, y_test = perform_feature_engineering(data_frame)

    train_or_load_models(X_train, X_test, y_train, y_test, args.train)
