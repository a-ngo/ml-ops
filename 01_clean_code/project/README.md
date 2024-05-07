# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project splits the `churn_notebook.ipynb` into a python library `churn_library.py` which can be used to train different models and run the predictions. Additionally, multiple evaluation and performance plots can be created.

## Files and data description
```
├── data/
│   └── bank_data.csv
├── images/
│   ├── churn_histogram.png
│   ├── heatmap.png
│   └── etc.
├── logs/
│   └── churn_library.log
├── models/
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── churn_library.py
├── churn_notebook.ipynb
├── Guide.ipynb
├── README.md
├── requirements.txt
└── churn_script_logging_and_tests.py
```

## Running Files
### Getting Started
You can install dependencies with:
```bash
python3 -m pip install -r requirements_py3.8.txt
```
### Run Unit Tests
First, see if you can run the unit tests with the following:
```shell
pytest churn_script_logging_and_tests.py
```

### Train models and run predictions
After that, you can run the script `execute` under Pipfile by issuing the following command on the terminal:
```shell
python3 churn_library.py
```

This should execute the `churn_library.py` script which loads and preprocess the data, creates EDA plots, trains the model and displays the ROC curve, among other things. The models can also be trained with the `--train` flag.
