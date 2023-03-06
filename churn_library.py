"""
This module holds several functions used to analyse and predict Customer Churn
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sns.set()

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    logging.info("file path = '%s'", pth)
    loaded_df = pd.read_csv(pth)
    logging.info("SUCCESS: Data imported successfully")
    return loaded_df


def perform_eda(loaded_df):
    '''
    perform eda on loaded_df and save figures to images folder
    input:
            loaded_df: pandas dataframe

    output:
            None
    '''
    loaded_df['Churn'] = loaded_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    churn_hist = loaded_df['Churn'].hist().get_figure()
    save_fig(churn_hist, 'images/churn_hist.png')
    age_hist = loaded_df['Customer_Age'].hist().get_figure()
    save_fig(age_hist, 'images/age_hist.png')
    logging.info("SUCCESS: hist_plot saved successfully.")
    marital_status_plot = loaded_df.Marital_Status.value_counts(
        'normalize').plot(kind='bar').get_figure()
    save_fig(marital_status_plot, 'images/marital_status_plot.png')
    logging.info("SUCCESS: marital status saved successfully.")
    total_trans_ct_plot = sns.histplot(
        loaded_df['Total_Trans_Ct'], stat='density', kde=True).get_figure()
    save_fig(total_trans_ct_plot, 'images/total_trans_ct_plot.png')
    logging.info("SUCCESS: Total_Trans_Ct histplot saved.")
    heatmap = sns.heatmap(loaded_df.corr(), annot=False, cmap='Dark2_r', linewidths=2).get_figure()
    save_fig(heatmap, 'images/heatmap.png')
    return loaded_df


def save_fig(plot, folder_path):
    '''
    Saves a plot into a folder.
    input:
            plt: AxesSubPlot
            folder_path: str

    output:
            None
    '''
    plot.savefig(folder_path)


def encoder_helper(loaded_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            loaded_df: pandas dataframe.
            category_lst: list of columns that contain categorical features.
            response: string of response name.

    output:
            df: pandas dataframe with new columns for
    '''
    encoder_df = loaded_df.copy(deep=True)
    for cat in category_lst:
        logging.info('Encoding column %s', cat)
        groups = loaded_df.groupby(cat).mean()['Churn']
        column_lst = [groups.loc[val] for val in loaded_df[cat]]
        if response:
            encoder_df[cat + '_' + response] = column_lst
        else:
            encoder_df[cat] = column_lst
    return encoder_df


def perform_feature_engineering(loaded_df, response):
    '''
    input:
              df: pandas dataframe.
              response: string of response name.

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    loaded_df = encoder_helper(loaded_df, cat_columns, response)
    feats = pd.DataFrame()
    label = loaded_df['Churn']

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

    feats[keep_cols] = loaded_df[keep_cols]
    feats_train, feats_test, label_train, label_test = train_test_split(
        feats, label, test_size=0.3, random_state=42)
    return feats_train, feats_test, label_train, label_test


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
    # RandomForestClassifier
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='images/rf_results.png')

    # LogisticRegression
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='images/logistic_results.png')


def feature_importance_plot(model, feat_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [feat_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(feat_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(feat_data.shape[1]), names, rotation=90)

    save_fig(plt, output_pth + "imporatance.png")


def train_models(feat_train, feat_test, label_train, label_test, output_pth):
    '''
    train, store model results: images + scores, and store models
    input:
              feat_train: X training data
              feat_test: X testing data
              label_train: y training data
              label_test: y testing data
              output_pth: the path to the output folder
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(feat_train, label_train)

    lrc.fit(feat_train, label_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(feat_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(feat_test)

    y_train_preds_lr = lrc.predict(feat_train)
    y_test_preds_lr = lrc.predict(feat_test)

    lrc_plot = RocCurveDisplay.from_estimator(lrc, feat_test, label_test)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_,
        feat_test,
        label_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)

    classification_report_image(label_train,
                                label_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc, feat_train, output_pth)


if __name__ == "__main__":
    print("It works!")
    df = import_data("./data/BankChurners.csv")
    perform_eda(df)
    df = encoder_helper(df, cat_columns, "Churn")
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")
    train_models(X_train, X_test, y_train, y_test, "output/")