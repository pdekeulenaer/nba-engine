import time, sys
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from scipy.special import logit, expit

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve
from sklearn.preprocessing import StandardScaler

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

# Standardize the data so that the mean is 0 and std dev is 1; integral will be 1
def standardize_data(df):
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    return df_standardized

# Score any type of model
# Flag on whether to use probability estimates or direct class predictions
# Model expects a "predict" and "predict_proba" function
def score_model(model, test_x, test_y, use_probability=True):
    if (use_probability): y = model.predict_proba(test_x)[:,1]  #TODO fix so it works for non binary stuff
    else: y = model.predict(test_x)
    score = roc_auc_score(test_y, y)

    return score

def main_boostclassifier(datastruct, n_estimators=50, base_min_samples_leaf=0.05, experiment_id=None):
    print("Starting experiment Ada Boosted Decision Trees")
    mlflow.set_experiment("Santander Kaggle")
    df, train_x, train_y, test_x, test_y = datastruct
    metrics = {}

    with mlflow.start_run():
        print("Training model")
        start_timer = time.time()

        # train 200 small models
        models = []
        for var in train_x.columns:
            sys.stdout.write('\r')
            base_estimator = DecisionTreeClassifier(min_samples_leaf=base_min_samples_leaf, random_state=0)
            booster = GradientBoostingClassifier(n_estimators=n_estimators, random_state=0)
            booster.fit(train_x[var].values.reshape(-1,1), train_y)
            models.append(booster)
            sys.stdout.write('> {} / 200'.format(len(models)))
            sys.stdout.flush()

        stop_timer = time.time()
        print ("Model trained")

        predictions = [m.predict_proba(x.reshape(-1,1))[:,1] for (m, x) in zip(models, test_x.values.T)]
        pred_y = np.array(predictions).T.mean(axis=1)
        pred_y_logit = logit(np.array(predictions).T).sum(axis=1)

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

def main_dtclassifier(datastruct, min_samples_leaf=0.1, experiment_id=None):
    print("Starting experiment Decision Trees")
    mlflow.set_experiment("Santander Kaggle")
    df, train_x, train_y, test_x, test_y = datastruct
    metrics = {}

    with mlflow.start_run():
        print("Training model")
        start_timer = time.time()

        # train 200 small models
        models = []
        for var in train_x.columns:
            clf = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=0)
            clf.fit(train_x[var].values.reshape(-1,1), train_y)
            models.append(clf)

        stop_timer = time.time()
        print ("Model trained")

        predictions = [m.predict_proba(x.reshape(-1,1))[:,1] for (m, x) in zip(models, test_x.values.T)]
        pred_y = np.array(predictions).T.mean(axis=1)
        pred_y_logit = logit(np.array(predictions).T).sum(axis=1)

        metrics['roc_auc'] = roc_auc_score(test_y, pred_y)
        metrics['roc_auc_logit'] = roc_auc_score(test_y, pred_y_logit)
        metrics['elapsed_time'] = (stop_timer - start_timer)


        #mlflow logging
        mlflow.log_param('model_type', "200 Decision Trees")
        mlflow.log_param('features', train_x.columns)
        mlflow.log_param('sample_size', df.shape)
        mlflow.log_param('min_samples_leaf', min_samples_leaf)
        mlflow.log_metrics(metrics)

        print ("Completed")

def main_nbclassifier(datastruct, experiment_id=None):
    print("Starting experiment NB")
    mlflow.set_experiment("Santander Kaggle")
    df, train_x, train_y, test_x, test_y = datastruct
    metrics = {}

    with mlflow.start_run():
        model = GaussianNB()
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


        metrics['roc_auc'] = score
        metrics['elapsed_time'] = (stop_timer - start_timer)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "Gaussian NB Classifier")
        print ("Completed")

def main_rfclassifier(n_est, max_depth, datastruct, experiment_id=None):
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
    print('> Loading data')
    datastruct = load_data(DATA_PATH, 'train.csv', sample_size=-1)
    #main_nbclassifier(datastruct)
    #for i in np.linspace(0.01,0.20,20):
    main_boostclassifier(datastruct, 50, 0.05)

    # # looping
    # for i in range(1,150,25):
    #     for j in range(2,16,4):
    #         main(i, j, datastruct)

#    for i in range(2,8,2):#
        #main_rfclassifier(80,i,datastruct)
