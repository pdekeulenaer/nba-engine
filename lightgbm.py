import time, sys
import mlflow
import pandas as pd
import numpy as np
from scipy.special import logit, expit

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve

import lightgbm

DATA_PATH = './data/santander-kaggle/'
EXPERIMENT_ID = None

def load_data(path, filename, test_size=0.1, sample_size=-1):
    df = pd.read_csv(path + filename)
    df = prepare_data(df)
    if (sample_size > -1): df = df.sample(sample_size)

    # split data in test_train
    train, test = train_test_split(df, test_size=test_size)
    train_x = train.drop(['target'], axis=1)
    train_y = train.target
    test_x = test.drop(['target'], axis=1)
    test_y = test.target
    return df, train_x, train_y, test_x, test_y

def prepare_data(df):
    df.index = df.ID_code
    return df.drop(['ID_code'], axis=1)

# Score any type of model
# Flag on whether to use probability estimates or direct class predictions
# Model expects a "predict" and "predict_proba" function
def score_model(model, test_x, test_y, use_probability=True):
    if (use_probability): y = model.predict_proba(test_x)[:,1]  #TODO fix so it works for non binary stuff
    else: y = model.predict(test_x)
    score = roc_auc_score(test_y, y)

    return score

def main_gbmclassifier(datastruct, experiment_id=None):
    print("Light GBM model")
    mlflow.set_experiment("Light GBM Experiments")
    df, train_x, train_y, test_x, test_y = datastruct

    train_data = lightgbm.Dataset(train_x, label=train_y)
    test_data = lightgbm.Dataset(test_x, label=test_y)

    metrics = {}

    with mlflow.start_run():
        print("Training model")
        start_timer = time.time()

        parameters = {
            'application': 'binary',
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': 'true'
        }

        lightgbm.train(parameters, train_data, valid_sets=test_data)
        pred_y = lightgbm.predict(test_x)

        # train 200 small models
        # models = []
        # for var in train_x.columns:
        #     sys.stdout.write('\r')
        #     #base_estimator = DecisionTreeClassifier(min_samples_leaf=base_min_samples_leaf, random_state=0)
        #     model = lightgbm.train(parameters, train_data, valid_sets=test_data)
        #     models.append(model)
        #     sys.stdout.write('> {} / 200'.format(len(models)))
        #     sys.stdout.flush()

        stop_timer = time.time()
        print ("Model trained")

        # predictions = [m.predict_proba(x.reshape(-1,1))[:,1] for (m, x) in zip(models, test_x.values.T)]

        # pred_y = np.array(predictions).T.mean(axis=1)
        # pred_y_logit = logit(np.array(predictions).T).sum(axis=1)

        metrics['roc_auc'] = roc_auc_score(test_y, pred_y)
        metrics['roc_auc_logit'] = roc_auc_score(test_y, pred_y_logit)
        metrics['elapsed_time'] = (stop_timer - start_timer)


        #mlflow logging
        mlflow.log_param('model_type', "200 Ada Boosted Decision Trees")
        mlflow.log_param('features', train_x.columns)
        mlflow.log_param('sample_size', df.shape)
        mlflow.log_param('min_samples_leaf', base_min_samples_leaf)
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_metrics(metrics)

        print ("Completed")

if __name__ == '__main__':
    print('> Loading data')
    datastruct = load_data(DATA_PATH, 'train.csv', sample_size=-1)
    #main_nbclassifier(datastruct)
    #for i in np.linspace(0.01,0.20,20):
    main_gbmclassifier(datastruct)

    # # looping
    # for i in range(1,150,25):
    #     for j in range(2,16,4):
    #         main(i, j, datastruct)

#    for i in range(2,8,2):#
        #main_rfclassifier(80,i,datastruct)
