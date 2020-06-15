import time
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve

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

def score_model(model, test_x, test_y, use_probability=True):
    if (use_probability): y = model.predict_proba(test_x)[:,1]  #TODO fix so it works for non binary stuff
    else: y = model.predict(test_x)
    score = roc_auc_score(test_y, y)

    return score

def main(n_est, max_depth, datastruct, experiment_id=None):
    print ("Starting experiment [{}, {}]".format(n_est, max_depth))
    df, train_x, train_y, test_x, test_y = datastruct
    metrics = {}

    # if no experiment, set it up
    print ("Setting up experiment")
    mlflow.set_experiment('RandomForest Classifier')

    with mlflow.start_run():
        # model params
        model = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, class_weight='balanced')

        print("Training model")
        # train the model
        start_timer = time.time()
        model.fit(train_x, train_y)
        stop_timer = time.time()
        print ("Model trained")

        score = score_model(model, test_x, test_y, True)

        #mlflow logging
        mlflow.log_param('model_type', str(model.__class__))
        mlflow.log_param('features', train_x.columns)
        mlflow.log_param('sample_size', df.shape)
        mlflow.log_params(model.get_params())

        metrics['roc_auc'] = score
        metrics['elapsed_time'] = (stop_timer - start_timer)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "Random Forest Classifier")
        print ("Completed")

if __name__ == '__main__':
    datastruct = load_data(DATA_PATH, 'train.csv', sample_size=-1)

    # looping
    for i in range(1,150,25):
        for j in range(2,16,4):
            main(i, j, datastruct)

    main()
