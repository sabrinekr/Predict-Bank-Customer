"""
This module tests the Churn Library functions.

Author: Sabrine Krichen
Date: Mar 2023
"""
import os
import logging
import pytest
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
               'Income_Category', 'Card_Category'
               ]
keep_columns = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
    'Income_Category_Churn', 'Card_Category_Churn']

eda_files = [
    'Customer_Age_hist.png', 'Total_Trans_Ct.png', 
    'Marital_Status.png', 'Correlation_heat_map.png', 
    'Churn_hist.png']


@pytest.fixture
def import_data():
    '''
    Import data pytest fixture.
    '''
    return cl.import_data('./data/BankChurners.csv')


@pytest.fixture
def perform_eda(import_data):
    '''
    Perform eda pytest fixture.
    '''
    data = import_data
    return cl.perform_eda(data)


@pytest.fixture
def prepare_df(import_data):
    '''
    Prepare dataframe pytest fixture.
    '''
    data = import_data
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data


@pytest.fixture
def encoder_helper(prepare_df):
    '''
    Encoder helper pytest fixture.
    '''
    data = prepare_df
    return cl.encoder_helper(data, cat_columns, 'Churn')


@pytest.fixture
def perform_feature_engineering(encoder_helper):
    '''
    Feature engineering pytest fixture.
    '''
    data = encoder_helper
    return cl.perform_feature_engineering(data, 'Churn')


@pytest.fixture
def train_models(perform_feature_engineering):
    '''
    Train models pytest fixture.
    '''
    datum = perform_feature_engineering
    return cl.train_models(*datum, "output")


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        loaded_df = cl.import_data('./data/BankChurners.csv')
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert loaded_df.shape[0] > 0
        assert loaded_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        _ = perform_eda
        assert eda_files == os.listdir('./images/')
        logging.info('Testing perform_eda: SUCCESS')
    except AssertionError:
        logging.error(
            'Testing perform_eda: Some expected figures were not found')


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        data = encoder_helper
        assert list(data.columns) == cat_columns
        logging.info('Testing encoder_helper: SUCCESS')
    except AssertionError:
        logging.error(
            'Testing encoder_helper: Some expected columns were not found')


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        df_train, df_test, _, _ = perform_feature_engineering
        assert list(df_train.columns) == keep_columns
        assert list(df_test.columns) == keep_columns
        logging.info('Test perform_feature_engineering: SUCCESS')
    except AssertionError:
        logging.error(
            'Test perform_feature_engineering: Unexpected features selection')


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        _ = train_models
        logging.info('Test train_models: SUCCESS')
        os.path.exists('./models/rfc_model.pkl')
        os.path.exists('./models/logistic_model.pkl')
    except BaseException:
        logging.exception('Test train_models: Unexpected Error')
