"""
Unit test of churn_library with
author: a-ngo
date: 2024-05-08
"""

import logging

import pytest

from churn_library import import_data, perform_eda, encoder_helper, \
    perform_feature_engineering, train_or_load_models

logging.basicConfig(
    filename='./logs/test_churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="module")
def data_frame():
    """
    Provide the data as a fixture to be used in the tests
    """
    return import_data("./data/bank_data.csv")


@pytest.fixture(scope="module")
def columns_to_encode():
    """
    Provide the columns to encode as a fixture to be used in the tests
    """
    return ["Gender", "Education_Level",
            "Income_Category", "Marital_Status", "Card_Category"]


def test_import(data_frame):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = data_frame
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] == 10127
        assert df.shape[1] == 23
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't have the expected row and column sizes!")
        raise err


def test_eda(data_frame):
    '''
    test perform eda function
    '''
    try:
        perform_eda(data_frame)
        logging.info("Testing eda: SUCCESS")
    except Exception as err:
        logging.error("Testing eda: An error occurred!")
        raise err


def test_encoder_helper(data_frame, columns_to_encode):
    '''
    test encoder helper
    '''
    df = encoder_helper(data_frame, columns_to_encode)

    try:
        assert df.shape[0] == 10127
        assert df.shape[1] == 28
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The file doesn't have the expected row and column sizes!")
        raise err


def test_perform_feature_engineering(data_frame):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            data_frame)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: An error occurred!")
        raise err

    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info(
            "Testing perform_feature_engineering resulting shapes: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: \
            The resulting data does not have the expected shapes!")
        raise err


def test_train_models(data_frame):
    '''
    test train_models
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        data_frame)
    try:
        train_or_load_models(X_train, X_test, y_train, y_test, True)
        logging.info("Testing training models: SUCCESS")
    except Exception as err:
        logging.error("Testing training models: An error occurred!")
        raise err

    try:
        train_or_load_models(X_train, X_test, y_train, y_test, False)
        logging.info("Testing model inference: SUCCESS")
    except Exception as err:
        logging.error("Testing model inference: An error occurred!")
        raise err


if __name__ == "__main__":
    pass
