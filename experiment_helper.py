import numpy as np
from random import shuffle
import pandas as pd
import random
# import pyfim
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from util import *
import matplotlib.pyplot as plt
import xgboost as xgb
import hyrs
from tr import *
from brs import *
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split as split
from copy import deepcopy
from scipy.stats import bernoulli, uniform
from sklearn import cluster, datasets
from scipy.io import arff
from sklearn.utils import shuffle
from scipy.stats import norm

xgb.set_config(verbosity=0)




def make_gaussians(numExamples=5000, numFeats=20):
    startDict = {}
    Xtrain = {}
    for feat in range(numFeats):
        Xtrain[feat] = norm.rvs(size=numExamples)

    startDict['Xtrain'] = pd.DataFrame(Xtrain)

    #humanGood make
    startDict['Xtrain']['humangood'] = 0
    pdfs = norm.pdf(np.sum(startDict['Xtrain'], axis=1))
    startDict['Xtrain']['humangood'][pdfs > np.median(pdfs)] = 1

    #accept make
    startDict['Xtrain']['accept'] = 0
    sums = np.sum(startDict['Xtrain'], axis=1)
    startDict['Xtrain']['accept'][sums < -0.5] = 1

    #makeLabels
    startDict['Ytrain'] = np.zeros(numExamples)
    startDict['Ytrain'][startDict['Xtrain']['accept'] == 1] = norm.pdf(np.sum(startDict['Xtrain'].iloc[:, 0:2][startDict['Xtrain']['accept'] == 1], axis=1))
    #startDict['Ytrain'][startDict['Xtrain']['accept'] == 1] = np.sum(startDict['Xtrain'].iloc[:, 0:2][startDict['Xtrain']['accept'] == 1], axis=1)
    startDict['Ytrain'][startDict['Xtrain']['accept'] == 1] = startDict['Ytrain'][startDict['Xtrain']['accept'] == 1] > np.median(startDict['Ytrain'][startDict['Xtrain']['accept'] == 1])

    startDict['Ytrain'][startDict['Xtrain']['accept'] == 0] = norm.pdf(np.sum(startDict['Xtrain'].iloc[:, 0:int(numFeats/4)][startDict['Xtrain']['accept'] == 0], axis=1)) + \
        norm.pdf(np.sum(startDict['Xtrain'].iloc[:, int(numFeats / 4):int(numFeats / 4)*2][startDict['Xtrain']['accept'] == 0], axis=1)) + \
        norm.pdf(np.sum(startDict['Xtrain'].iloc[:, int(numFeats / 4)*2:int(numFeats / 4)*4][startDict['Xtrain']['accept'] == 0], axis=1)) + \
        norm.pdf(np.sum(startDict['Xtrain'].iloc[:, int(numFeats / 4)*4:][startDict['Xtrain']['accept'] == 0], axis=1))

    startDict['Ytrain'][startDict['Xtrain']['accept'] == 0] = startDict['Ytrain'][
                                                                  startDict['Xtrain']['accept'] == 0] < np.median(
        startDict['Ytrain'][startDict['Xtrain']['accept'] == 0])

    startDict['Ytrain'] = pd.Series(startDict['Ytrain'])


    #shuffles
    startDict['Xtrain'], startDict['Ytrain'] = shuffle(startDict['Xtrain'], startDict['Ytrain'], random_state=0)
    startDict['Xtrain'] = startDict['Xtrain'].reset_index(drop=True)
    startDict['Ytrain'] = startDict['Ytrain'].reset_index(drop=True)

    # make test
    startDict['Xtest'] = startDict['Xtrain'].iloc[-500:, :].reset_index(drop=True)
    startDict['Xtrain'] = startDict['Xtrain'].iloc[:-500, :]
    startDict['Ytest'] = startDict['Ytrain'][-500:].reset_index(drop=True)
    startDict['Ytrain'] = startDict['Ytrain'][:-500]
    startDict['Xval'] = startDict['Xtest'].copy()
    startDict['Yval'] = startDict['Ytest'].copy()


    return startDict



def make_checkers(numExamples=5000):
    xy_min = [0, 0]
    xy_max = [2, 2]

    startDict = {}
    startDict['Xtrain'] = np.random.uniform(low=xy_min, high=xy_max, size=(numExamples, 2))
    startDict['Ytrain'] = np.zeros(numExamples)
    checkers = np.indices((2,2)).sum(axis=0) % 2

    for i in range(numExamples):
        startDict['Ytrain'][i] = checkers[int(startDict['Xtrain'][i][0]),
                                          int(startDict['Xtrain'][i][1])]



    startDict['Xtrain'] = pd.DataFrame({0: startDict['Xtrain'][:, 0],
                                        1: startDict['Xtrain'][:, 1]})
    startDict['Ytrain'] = pd.Series(startDict['Ytrain'])

    startDict['Xtrain']['humangood'] =  1
    startDict['Xtrain']['humangood'][startDict['Xtrain'][0] > startDict['Xtrain'][1]] = 0
    plt.scatter(startDict['Xtrain'][0][startDict['Ytrain'] == 1], startDict['Xtrain'][1][startDict['Ytrain'] == 1],
                c='r', s=0.2)
    plt.scatter(startDict['Xtrain'][0][startDict['Ytrain'] == 0], startDict['Xtrain'][1][startDict['Ytrain'] == 0],
                c='b', s=0.2)
    plt.show()

    startDict['Xtrain'], startDict['Ytrain'] = shuffle(startDict['Xtrain'], startDict['Ytrain'], random_state=0)
    startDict['Xtrain'] = startDict['Xtrain'].reset_index(drop=True)
    startDict['Ytrain'] = startDict['Ytrain'].reset_index(drop=True)

    # make test
    startDict['Xtest'] = startDict['Xtrain'].iloc[-500:, :].reset_index(drop=True)
    startDict['Xtrain'] = startDict['Xtrain'].iloc[:-500, :]
    startDict['Ytest'] = startDict['Ytrain'][-500:].reset_index(drop=True)
    startDict['Ytrain'] = startDict['Ytrain'][:-500]
    startDict['Xval'] = startDict['Xtest'].copy()
    startDict['Yval'] = startDict['Ytest'].copy()


    return startDict


def descretize_moons(startDict, numQs = 30):
    for col in startDict['Xtrain'].columns:
        for q in range(1,numQs):
            quantile = round((1/(numQs+1))*q,2)
            quantVal = round(np.quantile(startDict['Xtrain'][col], q=quantile),1)
            startDict['Xtrain'][str(col)+'_{}'.format(quantVal)] = (startDict['Xtrain'][col] > quantVal).astype(int)
            startDict['Xval'][str(col) + '_{}'.format(quantVal)] = (startDict['Xval'][col] > quantVal).astype(int)
            startDict['Xtest'][str(col) + '_{}'.format(quantVal)] = (startDict['Xtest'][col] > quantVal).astype(int)
        startDict['Xtrain'] = startDict['Xtrain'].drop(columns=[col])
        startDict['Xval'] = startDict['Xval'].drop(columns=[col])
        startDict['Xtest'] = startDict['Xtest'].drop(columns=[col])

    return startDict

def make_HR_data(numQs=5):
    startDict = {}
    data = arff.loadarff('Train-Natural-HR_employee_attrition.arff')
    startDict['Xtrain'] = pd.DataFrame(data[0])


    startDict['Xtrain'] = startDict['Xtrain'].sample(frac=1, random_state=1).reset_index(drop=True)

    startDict['Ytrain'] = startDict['Xtrain']['Attrition']
    startDict['Ytrain'] = startDict['Ytrain'].str.decode('UTF-8')
    startDict['Ytrain'] = startDict['Ytrain'].replace({"Yes": 1, "No": 0})
    startDict['Xtrain'].drop('Attrition', inplace=True, axis=1)

    str_df = startDict['Xtrain'].select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        startDict['Xtrain'][col] = str_df[col]

    categoricals = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'JobSatisfaction',
                    'MaritalStatus', 'Over18', 'OverTime']
    for col in startDict['Xtrain'].columns:
        if col not in categoricals:
            for q in range(1,numQs):
                quantile = round((1/(numQs+1))*q,2)
                quantVal = round(np.quantile(startDict['Xtrain'][col], q=quantile),1)
                startDict['Xtrain'][col+'{}'.format(quantVal)] = (startDict['Xtrain'][col] > quantVal).astype(int)
            startDict['Xtrain'] = startDict['Xtrain'].drop(columns=[col])


    for col in categoricals:
        startDict['Xtrain'] = pd.concat([startDict['Xtrain'],
                                     pd.get_dummies(startDict['Xtrain'][col],
                                                    prefix=col)], axis=1)

    startDict['Xtrain'] = startDict['Xtrain'].drop(
        columns=categoricals)


    # make test
    startDict['Xtest'] = startDict['Xtrain'].iloc[-90:, :]
    startDict['Xtrain'] = startDict['Xtrain'].iloc[:-90, :]
    startDict['Ytest'] = startDict['Ytrain'][-90:]
    startDict['Ytrain'] = startDict['Ytrain'][:-90]
    startDict['Xval'] = startDict['Xtest'].copy()
    startDict['Yval'] = startDict['Ytest'].copy()

    return startDict


def make_FICO_data(numQs=5):
    startDict = {}
    # train
    startDict['Xtrain'] = pd.read_csv('FICO.csv')
    startDict['Ytrain'] = startDict['Xtrain']['RiskPerformance']
    startDict['Ytrain'] = startDict['Ytrain'].replace({"Bad": 1, "Good":0})


    for col in startDict['Xtrain'].columns:
        if col not in ['RiskPerformance', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver']:
            for q in range(1,numQs):
                quantile = round((1/(numQs+1))*q,2)
                quantVal = round(np.quantile(startDict['Xtrain'][col], q=quantile),1)
                startDict['Xtrain'][col+'{}'.format(quantVal)] = (startDict['Xtrain'][col] > quantVal).astype(int)
            startDict['Xtrain'] = startDict['Xtrain'].drop(columns=[col])



    startDict['Xtrain'] = pd.concat([startDict['Xtrain'],
                                     pd.get_dummies(startDict['Xtrain'].MaxDelq2PublicRecLast12M, prefix='MaxDelq2PublicRecLast12M')], axis=1)
    startDict['Xtrain'] = pd.concat([startDict['Xtrain'],
                                     pd.get_dummies(startDict['Xtrain'].MaxDelqEver, prefix='MaxDelqEver')], axis=1)
    startDict['Xtrain'] = startDict['Xtrain'].drop(
        columns=['RiskPerformance', 'MaxDelqEver',  'MaxDelq2PublicRecLast12M'])

    #make test
    startDict['Xtest'] = startDict['Xtrain'].iloc[-500:,:]
    startDict['Xtrain'] = startDict['Xtrain'].iloc[:-500,:]
    startDict['Ytest'] = startDict['Ytrain'][-500:]
    startDict['Ytrain'] = startDict['Ytrain'][:-500]
    startDict['Xval'] = startDict['Xtest'].copy()
    startDict['Yval'] = startDict['Ytest'].copy()

    return startDict

def make_Adult_data():
    startDict = {}
    # train
    startDict['Xtrain'] = pd.read_csv('adult_train1.csv')
    startDict['Ytrain'] = startDict['Xtrain']['Y']

    startDict['Xtrain'] = pd.concat([startDict['Xtrain'],
                                     pd.get_dummies(startDict['Xtrain'].age, prefix='age')], axis=1)
    startDict['Xtrain'] = pd.concat([startDict['Xtrain'],
                                     pd.get_dummies(startDict['Xtrain'].educationnum, prefix='educationnum')], axis=1)
    startDict['Xtrain'] = pd.concat([startDict['Xtrain'],
                                     pd.get_dummies(startDict['Xtrain'].hoursperweek, prefix='hoursperweek')], axis=1)
    startDict['Xtrain'] = startDict['Xtrain'].drop(
        columns=['age', 'fnlwgt', 'educationnum', 'capitalgain', 'capitalloss',
                 'hoursperweek', 'Y', 'workclass_?'])
    # test
    startDict['Xtest'] = pd.read_csv('adult_test1.csv')
    # startDict['Xtest'] = startDict['Xtest'].sample(frac=1)
    startDict['Ytest'] = startDict['Xtest']['Y']

    startDict['Xtest'] = pd.concat([startDict['Xtest'], pd.get_dummies(startDict['Xtest'].age, prefix='age')], axis=1)
    startDict['Xtest'] = pd.concat(
        [startDict['Xtest'], pd.get_dummies(startDict['Xtest'].educationnum, prefix='educationnum')], axis=1)
    startDict['Xtest'] = pd.concat(
        [startDict['Xtest'], pd.get_dummies(startDict['Xtest'].hoursperweek, prefix='hoursperweek')], axis=1)

    startDict['Xtest'] = startDict['Xtest'].drop(columns=['age', 'fnlwgt', 'educationnum', 'capitalgain', 'capitalloss',
                                                          'hoursperweek', 'Y', 'workclass_?'])

    # test data did not have some values that training data had, just add these columns with all 0
    for item in np.setdiff1d(startDict['Xtrain'].columns, startDict['Xtest'].columns):
        startDict['Xtest'][item] = np.zeros(startDict['Xtest'].shape[0])

    startDict['Xtest'] = startDict['Xtest'][startDict['Xtrain'].columns]

    # split into val and test sets
    startDict['Xval'] = startDict['Xtest'].iloc[0:3000, :]
    startDict['Xtest'] = startDict['Xtest'].iloc[3000:, :]
    startDict['Yval'] = startDict['Ytest'].iloc[0:3000]
    startDict['Ytest'] = startDict['Ytest'].iloc[3000:]

    return startDict

def make_med_data():
    startDict = {}
    # train
    startDict['Xtrain'] = pd.read_csv('adult_train1.csv')
    startDict['Ytrain'] = startDict['Xtrain']['Y']

    startDict['Xtrain'] = pd.concat([startDict['Xtrain'],
                                     pd.get_dummies(startDict['Xtrain'].age, prefix='age')], axis=1)
    startDict['Xtrain'] = pd.concat([startDict['Xtrain'],
                                     pd.get_dummies(startDict['Xtrain'].educationnum, prefix='educationnum')], axis=1)
    startDict['Xtrain'] = pd.concat([startDict['Xtrain'],
                                     pd.get_dummies(startDict['Xtrain'].hoursperweek, prefix='hoursperweek')], axis=1)
    startDict['Xtrain'] = startDict['Xtrain'].drop(
        columns=['age', 'fnlwgt', 'educationnum', 'capitalgain', 'capitalloss',
                 'hoursperweek', 'Y', 'workclass_?'])
    # test
    startDict['Xtest'] = pd.read_csv('adult_test1.csv')
    # startDict['Xtest'] = startDict['Xtest'].sample(frac=1)
    startDict['Ytest'] = startDict['Xtest']['Y']

    startDict['Xtest'] = pd.concat([startDict['Xtest'], pd.get_dummies(startDict['Xtest'].age, prefix='age')], axis=1)
    startDict['Xtest'] = pd.concat(
        [startDict['Xtest'], pd.get_dummies(startDict['Xtest'].educationnum, prefix='educationnum')], axis=1)
    startDict['Xtest'] = pd.concat(
        [startDict['Xtest'], pd.get_dummies(startDict['Xtest'].hoursperweek, prefix='hoursperweek')], axis=1)

    startDict['Xtest'] = startDict['Xtest'].drop(columns=['age', 'fnlwgt', 'educationnum', 'capitalgain', 'capitalloss',
                                                          'hoursperweek', 'Y', 'workclass_?'])

    # test data did not have some values that training data had, just add these columns with all 0
    for item in np.setdiff1d(startDict['Xtrain'].columns, startDict['Xtest'].columns):
        startDict['Xtest'][item] = np.zeros(startDict['Xtest'].shape[0])

    startDict['Xtest'] = startDict['Xtest'][startDict['Xtrain'].columns]

    # split into val and test sets
    startDict['Xval'] = startDict['Xtest'].iloc[0:3000, :]
    startDict['Xtest'] = startDict['Xtest'].iloc[3000:, :]
    startDict['Yval'] = startDict['Ytest'].iloc[0:3000]
    startDict['Ytest'] = startDict['Ytest'].iloc[3000:]


class HAI_team():

    def __init__(self, data_model_dict):
        self.og_data_model_dict = deepcopy(data_model_dict)
        self.data_model_dict = deepcopy(data_model_dict)
        self.human = None
        self.mental_aversion = None
        self.mental_error_boundary = None
        self.auFair = None
        self.model = None
        self.results = None
        self.tr = None
        self.hyrs = None
        self.fA = None
        self.force_complete_coverage = False
        self.asym_loss = [1,1]
        self.asym_accept = 0

    def set_training_params(self, Niteration, Nchain, Nlevel, Nrules,
                            supp, maxlen, protected, budget, sample_ratio, alpha=0, beta=0, iters=1000, coverage_reg=0,
                            contradiction_reg=0, fA=0.5, rejectType='all', force_complete_coverage=False, asym_loss = [1,1], asym_accept = 0):
        self.Niteration = Niteration
        self.Nchain = Nchain
        self.Nlevel = Nlevel
        self.Nrules = Nrules
        self.supp = supp
        self.maxlen = maxlen
        self.protected = protected
        self.budget = budget
        self.sample_ratio = sample_ratio
        self.alpha = alpha
        self.beta = beta
        self.iters = iters
        self.coverage_reg = coverage_reg
        self.contradiction_reg = contradiction_reg
        self.fA = fA
        self.rejectType = rejectType
        self.force_complete_coverage = force_complete_coverage
        self.asym_loss = asym_loss
        self.asym_accept = asym_accept

    def make_human_model(self, type='logistic', acceptThreshold=0.5, numExamplesToUse=100, numColsToUse=0,
                         biasFactor=0, partial_train_percent = 1, alterations=None, drop=[]):


        self.data_model_dict = deepcopy(self.og_data_model_dict)

        self.accept_criteria = acceptThreshold
        if numColsToUse != 0:
            cols = []
            for i in range(len(self.data_model_dict['Xtrain'].columns)):
                cols.append(i)
            shuffle(cols)
            cols = cols[0:numColsToUse]
            #if list(self.data_model_dict['Xtrain'].columns).index(self.protected) not in cols:
            #    cols.append(list(self.data_model_dict['Xtrain'].columns).index(self.protected))
        else:
            cols = range(0, len(self.data_model_dict['Xtrain'].drop(columns=drop).columns))

        self.human_cols_used = cols

        if type == 'logistic':
            # simulate a 'True' human using a logistic regression model trained on very little data
            human_model = LogisticRegression(solver='sag', class_weight='balanced').fit(
                self.data_model_dict['Xtrain'].drop(columns=drop).iloc[0:round(partial_train_percent*numExamplesToUse), cols],
                self.data_model_dict['Ytrain'].iloc[0:round(partial_train_percent*numExamplesToUse)])

        elif type == 'xgboost':
            # simulate a 'True' human using a logistic regression model trained on very little data
            human_model = xgb.XGBClassifier()
            human_model.fit(
                self.data_model_dict['Xtrain'].drop(columns=drop).iloc[0:round(partial_train_percent*numExamplesToUse), cols],
                self.data_model_dict['Ytrain'].iloc[0:round(partial_train_percent*numExamplesToUse)])

        self.data_model_dict['Xtrain'] = self.data_model_dict['Xtrain'].iloc[numExamplesToUse:, :].reset_index(
            drop=True)  # dont use data that trained human going forward
        self.data_model_dict['Ytrain'] = self.data_model_dict['Ytrain'].iloc[numExamplesToUse:].reset_index(
            drop=True)  # dont use data that trained human going forward
        #print('Accuracy of Human on Training Data: ' + str(
        #human_model.score(self.data_model_dict['Xtrain'].iloc[:, cols], self.data_model_dict['Ytrain'])))
        Ybtrain = human_model.predict(self.data_model_dict['Xtrain'].drop(columns=drop).iloc[:, cols])  # human prediction
        conf = (human_model.predict_proba(self.data_model_dict['Xtrain'].drop(columns=drop).iloc[:, cols]))
        conf = np.abs(
            conf[:, 0] - 0.5) * 2  # human confidence level, will be used with a threshold to determine acceptance

        

        self.human = human_model
        self.data_model_dict['Ybtrain'] = Ybtrain
        self.data_model_dict['train_conf'] = conf
        self.data_model_dict['train_accept'] = pd.Series(conf < acceptThreshold).reset_index(drop=True)
        self.data_model_dict['Ybtest'] = human_model.predict(self.data_model_dict['Xtest'].iloc[:, self.human_cols_used])
        test_conf = (self.human.predict_proba(self.data_model_dict['Xtest'].iloc[:, self.human_cols_used]))
        test_conf = np.abs(test_conf[:, 0] - 0.5) * 2  # true human's confidence for new cases

        self.data_model_dict['test_conf'] = test_conf
        self.data_model_dict['test_accept'] = pd.Series(test_conf < acceptThreshold).reset_index(drop=True)

        self.data_model_dict['Ybval'] = human_model.predict(self.data_model_dict['Xval'].iloc[:, cols])
        val_conf = (self.human.predict_proba(self.data_model_dict['Xval'].iloc[:, cols]))
        val_conf = np.abs(val_conf[:, 0] - 0.5) * 2  # true human's confidence for new cases
        

        self.data_model_dict['val_conf'] = val_conf
        self.data_model_dict['val_accept'] = pd.Series(val_conf < acceptThreshold).reset_index(drop=True)

        if alterations != None:

            #increase train accuracy at good region to goodProb

            trainCorrections = bernoulli.rvs(p=alterations['goodProb'], size=len(self.data_model_dict['Ybtrain'][(self.data_model_dict['train_conf'] > alterations['goodRange'][0]) \
                                                                                                                 & (self.data_model_dict['train_conf'] <= alterations['goodRange'][1])])).astype(bool)
            self.data_model_dict['Ybtrain'][np.where((self.data_model_dict['train_conf'] > alterations['goodRange'][0]) \
                                                     & (self.data_model_dict['train_conf'] <= alterations['goodRange'][1]))[0][np.where(trainCorrections)[0]]] = self.data_model_dict['Ytrain'][(self.data_model_dict['train_conf'] > alterations['goodRange'][0]) \
                                                                                                                                                                                                & (self.data_model_dict['train_conf'] <= alterations['goodRange'][1])][trainCorrections]
            self.data_model_dict['Ybtrain'][np.where((self.data_model_dict['train_conf'] > alterations['goodRange'][0]) \
                                                     & (self.data_model_dict['train_conf'] <= alterations['goodRange'][1]))[0][np.where(~trainCorrections)[0]]] = \
                np.abs(1-self.data_model_dict['Ytrain'][(self.data_model_dict['train_conf'] > alterations['goodRange'][0]) \
                                                        & (self.data_model_dict['train_conf'] <= alterations['goodRange'][1])][
                    ~trainCorrections])

            #decrease train accuracy at bad region to badProb
            trainInCorrections = bernoulli.rvs(p=alterations['badProb'], size=len(
                self.data_model_dict['Ybtrain'][(self.data_model_dict['train_conf'] > alterations['badRange'][0]) \
                                                & (self.data_model_dict['train_conf'] <= alterations['badRange'][
                    1])])).astype(bool)
            self.data_model_dict['Ybtrain'][np.where((self.data_model_dict['train_conf'] > alterations['badRange'][0]) \
                                                     & (self.data_model_dict['train_conf'] <= alterations['badRange'][1]))[0][
                np.where(trainInCorrections)[0]]] = \
                self.data_model_dict['Ytrain'][(self.data_model_dict['train_conf'] > alterations['badRange'][0]) \
                                               & (self.data_model_dict['train_conf'] <= alterations['badRange'][1])][
                    trainInCorrections]
            self.data_model_dict['Ybtrain'][np.where((self.data_model_dict['train_conf'] > alterations['badRange'][0]) \
                                                     & (self.data_model_dict['train_conf'] <= alterations['badRange'][1]))[0][
                np.where(~trainInCorrections)[0]]] = \
                np.abs(1 - self.data_model_dict['Ytrain'][
                    (self.data_model_dict['train_conf'] > alterations['badRange'][0]) \
                    & (self.data_model_dict['train_conf'] <= alterations['badRange'][1])][
                    ~trainInCorrections])

            # increase val accuracy at good region to goodProb

            valCorrections = bernoulli.rvs(p=alterations['goodProb'], size=len(
                self.data_model_dict['Ybval'][(self.data_model_dict['val_conf'] > alterations['goodRange'][0]) \
                                              & (self.data_model_dict['val_conf'] <= alterations['goodRange'][
                    1])])).astype(bool)
            self.data_model_dict['Ybval'][np.where((self.data_model_dict['val_conf'] > alterations['goodRange'][0]) \
                                                   & (self.data_model_dict['val_conf'] <= alterations['goodRange'][
                1]))[0][np.where(valCorrections)[0]]] = \
                self.data_model_dict['Yval'][(self.data_model_dict['val_conf'] > alterations['goodRange'][0]) \
                                             & (self.data_model_dict['val_conf'] <= alterations['goodRange'][1])][
                    valCorrections]
            self.data_model_dict['Ybval'][np.where((self.data_model_dict['val_conf'] > alterations['goodRange'][0]) \
                                                   & (self.data_model_dict['val_conf'] <= alterations['goodRange'][
                1]))[0][np.where(~valCorrections)[0]]] = \
                np.abs(1 - self.data_model_dict['Yval'][
                    (self.data_model_dict['val_conf'] > alterations['goodRange'][0]) \
                    & (self.data_model_dict['val_conf'] <= alterations['goodRange'][1])][
                    ~valCorrections])

            # decrease val accuracy at bad region to badProb
            valInCorrections = bernoulli.rvs(p=alterations['badProb'], size=len(
                self.data_model_dict['Ybval'][(self.data_model_dict['val_conf'] > alterations['badRange'][0]) \
                                              & (self.data_model_dict['val_conf'] <= alterations['badRange'][
                    1])])).astype(bool)
            self.data_model_dict['Ybval'][np.where((self.data_model_dict['val_conf'] > alterations['badRange'][0]) \
                                                   & (self.data_model_dict['val_conf'] <= alterations['badRange'][
                1]))[0][
                np.where(valInCorrections)[0]]] = \
                self.data_model_dict['Yval'][(self.data_model_dict['val_conf'] > alterations['badRange'][0]) \
                                             & (self.data_model_dict['val_conf'] <= alterations['badRange'][1])][
                    valInCorrections]
            self.data_model_dict['Ybval'][np.where((self.data_model_dict['val_conf'] > alterations['badRange'][0]) \
                                                   & (self.data_model_dict['val_conf'] <= alterations['badRange'][
                1]))[0][
                np.where(~valInCorrections)[0]]] = \
                np.abs(1 - self.data_model_dict['Yval'][
                    (self.data_model_dict['val_conf'] > alterations['badRange'][0]) \
                    & (self.data_model_dict['val_conf'] <= alterations['badRange'][1])][
                    ~valInCorrections])

            # increase test accuracy at good region to goodProb

            testCorrections = bernoulli.rvs(p=alterations['goodProb'], size=len(
                self.data_model_dict['Ybtest'][(self.data_model_dict['test_conf'] > alterations['goodRange'][0]) \
                                               & (self.data_model_dict['test_conf'] <= alterations['goodRange'][
                    1])])).astype(bool)
            self.data_model_dict['Ybtest'][np.where((self.data_model_dict['test_conf'] > alterations['goodRange'][0]) \
                                                    & (self.data_model_dict['test_conf'] <= alterations['goodRange'][
                1]))[0][np.where(testCorrections)[0]]] = \
                self.data_model_dict['Ytest'][(self.data_model_dict['test_conf'] > alterations['goodRange'][0]) \
                                              & (self.data_model_dict['test_conf'] <= alterations['goodRange'][1])][
                    testCorrections]
            self.data_model_dict['Ybtest'][np.where((self.data_model_dict['test_conf'] > alterations['goodRange'][0]) \
                                                    & (self.data_model_dict['test_conf'] <= alterations['goodRange'][
                1]))[0][np.where(~testCorrections)[0]]] = \
                np.abs(1 - self.data_model_dict['Ytest'][
                    (self.data_model_dict['test_conf'] > alterations['goodRange'][0]) \
                    & (self.data_model_dict['test_conf'] <= alterations['goodRange'][1])][
                    ~testCorrections])

            # decrease test accuracy at bad region to badProb
            testInCorrections = bernoulli.rvs(p=alterations['badProb'], size=len(
                self.data_model_dict['Ybtest'][(self.data_model_dict['test_conf'] > alterations['badRange'][0]) \
                                               & (self.data_model_dict['test_conf'] <= alterations['badRange'][
                    1])])).astype(bool)
            self.data_model_dict['Ybtest'][np.where((self.data_model_dict['test_conf'] > alterations['badRange'][0]) \
                                                    & (self.data_model_dict['test_conf'] <= alterations['badRange'][
                1]))[0][
                np.where(testInCorrections)[0]]] = \
                self.data_model_dict['Ytest'][(self.data_model_dict['test_conf'] > alterations['badRange'][0]) \
                                              & (self.data_model_dict['test_conf'] <= alterations['badRange'][1])][
                    testInCorrections]
            self.data_model_dict['Ybtest'][np.where((self.data_model_dict['test_conf'] > alterations['badRange'][0]) \
                                                    & (self.data_model_dict['test_conf'] <= alterations['badRange'][
                1]))[0][
                np.where(~testInCorrections)[0]]] = \
                np.abs(1 - self.data_model_dict['Ytest'][
                    (self.data_model_dict['test_conf'] > alterations['badRange'][0]) \
                    & (self.data_model_dict['test_conf'] <= alterations['badRange'][1])][
                    ~testInCorrections])





            #alter confidences
            if alterations['Rational']:
                adder = alterations['adder']
            else:
                adder = -alterations['adder']

            og_train_conf = self.data_model_dict['train_conf'].copy()
            og_val_conf = self.data_model_dict['val_conf'].copy()
            og_test_conf = self.data_model_dict['test_conf'].copy()


            self.data_model_dict['train_conf'][np.where((og_train_conf > alterations['goodRange'][0]) \
                                                        & (og_train_conf <= alterations['goodRange'][1]))[0]] = og_train_conf[np.where((og_train_conf > alterations['goodRange'][0]) \
                                                                                                                                       & (og_train_conf <= alterations['goodRange'][1]))[0]] + adder

            self.data_model_dict['train_conf'][np.where((og_train_conf > alterations['badRange'][0]) \
                                                        & (og_train_conf <= alterations[
                'badRange'][1]))[0]] = og_train_conf[np.where((og_train_conf > alterations['badRange'][0]) \
                                                              & (og_train_conf <= alterations[
                'badRange'][1]))[0]] - adder

            self.data_model_dict['val_conf'][np.where((og_val_conf > alterations['goodRange'][0]) \
                                                      & (og_val_conf <= alterations[
                'goodRange'][1]))[0]] = og_val_conf[np.where((og_val_conf > alterations['goodRange'][0]) \
                                                             & (og_val_conf <= alterations[
                'goodRange'][1]))[0]] + adder

            self.data_model_dict['val_conf'][np.where((og_val_conf > alterations['badRange'][0]) \
                                                      & (og_val_conf <= alterations[
                'badRange'][1]))[0]] = og_val_conf[np.where((og_val_conf > alterations['badRange'][0]) \
                                                            & (og_val_conf <= alterations[
                'badRange'][1]))[0]] - adder

            self.data_model_dict['test_conf'][np.where((og_test_conf > alterations['goodRange'][0]) \
                                                       & (og_test_conf <= alterations[
                'goodRange'][1]))[0]] = og_test_conf[np.where((og_test_conf > alterations['goodRange'][0]) \
                                                              & (og_test_conf <= alterations[
                'goodRange'][1]))[0]] + adder

            self.data_model_dict['test_conf'][np.where((og_test_conf > alterations['badRange'][0]) \
                                                       & (og_test_conf <= alterations[
                'badRange'][1]))[0]] = og_test_conf[np.where((og_test_conf > alterations['badRange'][0]) \
                                                             & (og_test_conf <= alterations[
                'badRange'][1]))[0]] - adder

            self.data_model_dict['train_accept'] = pd.Series(self.data_model_dict['train_conf'] < acceptThreshold).reset_index(drop=True)
            self.data_model_dict['val_accept'] = pd.Series(self.data_model_dict['val_conf'] < acceptThreshold).reset_index(drop=True)
            self.data_model_dict['test_accept'] = pd.Series(self.data_model_dict['test_conf'] < acceptThreshold).reset_index(drop=True)

        print('Accuracy of Human on Test Data: ' + str(
            human_model.score(self.data_model_dict['Xtest'].iloc[:, cols], self.data_model_dict['Ytest'])))

        self.data_model_dict['human_wrong_train'] = self.data_model_dict['Ytrain'] != self.data_model_dict['Ybtrain']
        self.data_model_dict['human_wrong_val'] = self.data_model_dict['Yval'] != self.data_model_dict['Ybval']
        self.data_model_dict['human_wrong_test'] = self.data_model_dict['Ytest'] != self.data_model_dict['Ybtest']

        self.data_model_dict['Ytrain'] == pd.Series(self.data_model_dict['Ytrain'])
        self.data_model_dict['Yval'] == pd.Series(self.data_model_dict['Yval'])
        self.data_model_dict['Ytest'] == pd.Series(self.data_model_dict['Ytest'])

        self.data_model_dict['Ybtrain'] == pd.Series(self.data_model_dict['Ybtrain'])

        self.data_model_dict['Ybval'] == pd.Series(self.data_model_dict['Ybval'])
        self.data_model_dict['Ybtest'] == pd.Series(self.data_model_dict['Ybtest'])



        self.post_human_dict = deepcopy(self.data_model_dict)

    def makeHumanGood(self):
        self.data_model_dict['Ybtrain'][np.where(self.data_model_dict['Xtrain']['humangood'])[0]] = \
            self.data_model_dict['Ytrain'][np.where(self.data_model_dict['Xtrain']['humangood'])[0]]

        self.data_model_dict['Ybval'][np.where(self.data_model_dict['Xval']['humangood'])[0]] = \
            self.data_model_dict['Yval'][np.where(self.data_model_dict['Xval']['humangood'])[0]]

        self.data_model_dict['Ybtest'][np.where(self.data_model_dict['Xtest']['humangood'])[0]] = \
            self.data_model_dict['Ytest'][np.where(self.data_model_dict['Xtest']['humangood'])[0]]
    
    
    def adjustConfidences(self, where_col=None, where_val=None, comp=None):
        # make all conf high (greater than 0.5)
        self.data_model_dict['train_conf'] = uniform.rvs(0.5, 0.45, size=len(
            self.data_model_dict['train_conf']))

        self.data_model_dict['val_conf'] = uniform.rvs(0.5, 0.45, size=len(
            self.data_model_dict['val_conf']))

        self.data_model_dict['test_conf'] = uniform.rvs(0.5, 0.45, size=len(
            self.data_model_dict['test_conf']))


        if comp=='negate':
            # make desired region low conf (less than 0.5)
            self.data_model_dict['train_conf'][np.where(self.data_model_dict['Xtrain'][where_col] <= where_val)[0]] = \
                uniform.rvs(0, 0.49, size=len(
                    self.data_model_dict['train_conf'][
                        np.where(self.data_model_dict['Xtrain'][where_col] <= where_val)[0]]))

            self.data_model_dict['val_conf'][np.where(self.data_model_dict['Xval'][where_col] <= where_val)[0]] = \
                uniform.rvs(0, 0.49, size=len(
                    self.data_model_dict['val_conf'][
                        np.where(self.data_model_dict['Xval'][where_col] <= where_val)[0]]))

            self.data_model_dict['test_conf'][np.where(self.data_model_dict['Xtest'][where_col] <= where_val)[0]] = \
                uniform.rvs(0, 0.49, size=len(
                    self.data_model_dict['test_conf'][
                        np.where(self.data_model_dict['Xtest'][where_col] <= where_val)[0]]))
            
        else:
            # make desired region low conf (less than 0.5)
            self.data_model_dict['train_conf'][np.where(self.data_model_dict['Xtrain'][where_col] >= where_val)[0]] = \
                uniform.rvs(0, 0.49, size=len(
                    self.data_model_dict['train_conf'][
                        np.where(self.data_model_dict['Xtrain'][where_col] >= where_val)[0]]))

            self.data_model_dict['val_conf'][np.where(self.data_model_dict['Xval'][where_col] >= where_val)[0]] = \
                uniform.rvs(0, 0.49, size=len(
                    self.data_model_dict['val_conf'][
                        np.where(self.data_model_dict['Xval'][where_col] >= where_val)[0]]))

            self.data_model_dict['test_conf'][np.where(self.data_model_dict['Xtest'][where_col] >= where_val)[0]] = \
                uniform.rvs(0, 0.49, size=len(
                    self.data_model_dict['test_conf'][
                        np.where(self.data_model_dict['Xtest'][where_col] >= where_val)[0]]))


    def makeAdditionalTestSplit(self, testPercent=0.1, replaceExisting=True, random_state=1, others = []):
        
        others.append(self)
        for team in others:

            team.data_model_dict = deepcopy(team.post_human_dict)

            if replaceExisting:
                team.data_model_dict['Xtest2'] = team.data_model_dict['Xtest'].copy()
                team.data_model_dict['Ytest2'] = team.data_model_dict['Ytest'].copy()
                team.data_model_dict['Ybtest2'] = team.data_model_dict['Ybtest'].copy()
                team.data_model_dict['test2_conf'] = team.data_model_dict['test_conf'].copy()
                team.data_model_dict['test2_accept'] = team.data_model_dict['test_accept'].copy()
                team.data_model_dict['human_wrong_test2'] = team.data_model_dict['human_wrong_test'].copy()


                team.data_model_dict['Xtrain'], team.data_model_dict['Xtest'], team.data_model_dict['Ytrain'], \
                team.data_model_dict['Ytest'] = split(team.data_model_dict['Xtrain'],
                                                      team.data_model_dict['Ytrain'],
                                                      test_size=testPercent,
                                                      stratify=team.data_model_dict['Ytrain'],
                                                      random_state=random_state)

                trainDex = team.data_model_dict['Xtrain'].index
                test2Dex = team.data_model_dict['Xtest'].index

                team.testDex = test2Dex

                team.data_model_dict['Ybtest'] = team.data_model_dict['Ybtrain'][test2Dex]
                team.data_model_dict['test_conf'] = pd.Series(team.data_model_dict['train_conf'][test2Dex]).reset_index(drop=True)
                team.data_model_dict['test_accept'] = team.data_model_dict['train_accept'][test2Dex].reset_index(drop=True)
                team.data_model_dict['human_wrong_test'] = team.data_model_dict['human_wrong_train'][test2Dex].reset_index(
                    drop=True)


                team.data_model_dict['Ybtrain'] = team.data_model_dict['Ybtrain'][trainDex]
                team.data_model_dict['train_conf'] = pd.Series(team.data_model_dict['train_conf'][trainDex]).reset_index(drop=True)
                team.data_model_dict['train_accept'] = team.data_model_dict['train_accept'][trainDex].reset_index(drop=True)
                team.data_model_dict['human_wrong_train'] = team.data_model_dict['human_wrong_train'][trainDex].reset_index(
                    drop=True)

                team.data_model_dict['Xtrain'] = team.data_model_dict['Xtrain'].reset_index(drop=True)
                team.data_model_dict['Xtest'] = team.data_model_dict['Xtest'].reset_index(drop=True)
                team.data_model_dict['Ytrain'] = team.data_model_dict['Ytrain'].reset_index(drop=True)
                team.data_model_dict['Ytest'] = team.data_model_dict['Ytest'].reset_index(drop=True)



            else:
                team.data_model_dict['Xtrain'],
                team.data_model_dict['Xtest2'],
                team.data_model_dict['Ytrain'],
                team.data_model_dict['Ytest2'] = split(team.data_model_dict['Xtrain'],
                                                       team.data_model_dict['Ytrain'],
                                                       test_size=testPercent,
                                                       stratify=team.data_model_dict['Ytrain'],
                                                       random_state=random_state)

                trainDex = team.data_model_dict['Xtrain'].index
                test2Dex = team.data_model_dict['Xtest2'].index

                team.data_model_dict['Ybtest2'] = team.data_model_dict['Ybtrain'][test2Dex]
                team.data_model_dict['test2_conf'] = team.data_model_dict['train_conf'][test2Dex]
                team.data_model_dict['test2_accept'] = team.data_model_dict['train_accept'][test2Dex]
                team.data_model_dict['human_wrong_test2'] = team.data_model_dict['human_wrong_train'][test2Dex]

                team.data_model_dict['Ybtrain'] = team.data_model_dict['Ybtrain'][trainDex]
                team.data_model_dict['train_conf'] = team.data_model_dict['train_conf'][trainDex]
                team.data_model_dict['train_accept'] = team.data_model_dict['train_accept'][trainDex]
                team.data_model_dict['human_wrong_train'] = team.data_model_dict['human_wrong_train'][trainDex]




    def set_custom_confidence(self, train, val, test, type='prob'):
        self.data_model_dict['train_conf'] = pd.Series(train).clip(lower=0,upper=1)
        self.data_model_dict['val_conf'] = pd.Series(val).clip(lower=0,upper=1)
        self.data_model_dict['test_conf'] = pd.Series(test).clip(lower=0,upper=1)

        if type=='prob':

            self.data_model_dict['train_accept'] = ~(pd.Series(bernoulli.rvs(p=train, size=len(train))).astype(bool))
            self.data_model_dict['val_accept'] = ~(pd.Series(bernoulli.rvs(p=val, size=len(val))).astype(bool))
            self.data_model_dict['test_accept'] = ~(pd.Series(bernoulli.rvs(p=test, size=len(test))).astype(bool))

        else:
            self.data_model_dict['train_accept'] = pd.Series(train < 0.5).astype(bool)
            self.data_model_dict['val_accept'] = pd.Series(val < 0.5).astype(bool)
            self.data_model_dict['test_accept'] = pd.Series(test < 0.5).astype(bool)


        self.post_human_dict = deepcopy(self.data_model_dict)

    def train_mental_aversion_model(self, type='xgboost', probWrong=0, noise=0):
        '''Needs to have human values before can be run'''

        if type == 'xgboost':
            self.mental_aversion = xgb.XGBClassifier()
            self.mental_aversion.fit(self.data_model_dict['Xtrain'], self.data_model_dict['train_accept'])
        elif type == 'logistic':
            self.mental_aversion = LogisticRegression(solver='sag').fit(self.data_model_dict['Xtrain'],
                                                                        self.data_model_dict['train_accept'])
        if type != 'perfect':
            self.data_model_dict['paccept_train'] = self.mental_aversion.predict_proba(self.data_model_dict['Xtrain'])[
                                                    :, 1]
            self.data_model_dict['paccept_val'] = self.mental_aversion.predict_proba(self.data_model_dict['Xval'])[:, 1]
            self.data_model_dict['paccept_test'] = self.mental_aversion.predict_proba(self.data_model_dict['Xtest'])[:,
                                                   1]
            print('Accuracy of Mental Aversion Model on Train Data: ' + str(
                self.mental_aversion.score(self.data_model_dict['Xtrain'], self.data_model_dict['train_accept'])))
            print('Accuracy of Mental Aversion Model on Val Data: ' + str(
                self.mental_aversion.score(self.data_model_dict['Xval'], self.data_model_dict['val_accept'])))

        elif type == 'perfect':
            trainVars = bernoulli.rvs(p=probWrong, size=len(self.data_model_dict['train_accept']))
            valVars = bernoulli.rvs(p=probWrong, size=len(self.data_model_dict['val_accept']))
            testVars = bernoulli.rvs(p=probWrong, size=len(self.data_model_dict['test_accept']))
            self.data_model_dict['paccept_train'] = self.data_model_dict['train_accept'].copy()
            self.data_model_dict['paccept_train'][np.where(trainVars)[0]] = np.abs(1-self.data_model_dict['paccept_train'][np.where(trainVars)[0]])
            self.data_model_dict['paccept_train'] = (self.data_model_dict['paccept_train'] + uniform.rvs(-noise, 2*noise, size=len(self.data_model_dict['paccept_train']))).clip(lower=0, upper=1)
            self.data_model_dict['paccept_val'] = self.data_model_dict['val_accept'].reset_index(drop=True)
            self.data_model_dict['paccept_val'][np.where(valVars)[0]] = np.abs(1-self.data_model_dict['paccept_val'][np.where(valVars)[0]])
            self.data_model_dict['paccept_val'] = (
                        self.data_model_dict['paccept_val'] + uniform.rvs(-noise, 2 * noise, size=len(
                    self.data_model_dict['paccept_val']))).clip(lower=0, upper=1)
            self.data_model_dict['paccept_test'] = self.data_model_dict['test_accept'].reset_index(drop=True)
            self.data_model_dict['paccept_test'][np.where(testVars)[0]] = np.abs(1-self.data_model_dict['paccept_test'][np.where(testVars)[0]])
            self.data_model_dict['paccept_test'] = (
                        self.data_model_dict['paccept_test'] + uniform.rvs(-noise, 2 * noise, size=len(
                    self.data_model_dict['paccept_test']))).clip(lower=0, upper=1)
            
            print('Accuracy of Mental Aversion Model on Train Data: ' + str(
                metrics.accuracy_score(self.data_model_dict['paccept_train'] > 0.5,
                                       self.data_model_dict['train_accept'])))
            print('Accuracy of Mental Aversion Model on Val Data: ' + str(
                metrics.accuracy_score(self.data_model_dict['paccept_val'] > 0.5, self.data_model_dict['val_accept'])))

    def train_mental_error_boundary_model(self, type='xgboost'):
        #Needs to have human values before can be run
        if type == 'xgboost':
            self.mental_error_boundary = xgb.XGBClassifier()
            self.mental_error_boundary.fit(self.data_model_dict['Xtrain'], self.data_model_dict['human_wrong_train'])
        elif type == 'logistic':
            self.mental_error_boundary = LogisticRegression(solver='sag').fit(self.data_model_dict['Xtrain'],
                                                                              self.data_model_dict['human_wrong_train'])

        self.data_model_dict['prob_human_wrong_train'] = self.mental_error_boundary.predict_proba(
            self.data_model_dict['Xtrain'])[:, 1]
        self.data_model_dict['prob_human_wrong_val'] = self.mental_error_boundary.predict_proba(
            self.data_model_dict['Xval'])[:, 1]
        self.data_model_dict['prob_human_wrong_test'] = self.mental_error_boundary.predict_proba(
            self.data_model_dict['Xtest'])[:, 1]

        self.data_model_dict['pred_human_wrong_train'] = self.mental_error_boundary.predict(
            self.data_model_dict['Xtrain'])
        self.data_model_dict['pred_human_wrong_val'] = self.mental_error_boundary.predict(self.data_model_dict['Xval'])
        self.data_model_dict['pred_human_wrong_test'] = self.mental_error_boundary.predict(
            self.data_model_dict['Xtest'])
        print('Accuracy of Mental Error Boundary Model on Train Data: ' + str(
            self.mental_error_boundary.score(self.data_model_dict['Xtrain'],
                                             self.data_model_dict['human_wrong_train'])))
        print('Accuracy of Mental Error Boundary Model on Val Data: ' + str(
            self.mental_error_boundary.score(self.data_model_dict['Xval'], self.data_model_dict['human_wrong_val'])))


    def filter_hyrs_results(self, mental=False, error=False):
        #disregard filtering, it is related to other functionality we are working on and not to the paper, when set to false it should not filter
        if mental:
            mental_confs = [0, 0.0000001, 0.00001, 0.000025, 0.0001, 0.0005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.999, 0.9999,
                            0.99999,
                            0.999999, 0.9999999]
        else:
            mental_confs = [0]

        if error:
            error_confs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            error_confs = [0]

        full_hyrs_results = pd.DataFrame(index=range(len(mental_confs) * len(error_confs)),
                                             columns=['mental_conf',
                                                      'error_conf',
                                                      'train_covereds',
                                                      'modelonly_train_preds',
                                                      'humanified_train_preds',
                                                      'train_coverage',
                                                      'test_coverage',
                                                      'test_rejects',
                                                      'test_rejectsINC',
                                                      'test_rejectsCOR',
                                                      'test_error',
                                                      'test_covereds',
                                                      'modelonly_test_preds',
                                                      'humanified_test_preds'])
        index = 0
        for mental_conf in mental_confs:
            for error_conf in error_confs:
                full_hyrs_result = {}

                full_hyrs_preds = self.hyrs_results['modelonly_test_preds'].copy()
                full_hyrs_result['modelonly_test_preds'] = full_hyrs_preds.copy()

                full_hyrs_result['test_rejects'] = sum((self.data_model_dict['Ybtest'] != full_hyrs_preds)[
                                                               (self.data_model_dict['paccept_test'] >= mental_conf) & ~
                                                               self.data_model_dict['test_accept'] &
                                                               (self.data_model_dict[
                                                                    'prob_human_wrong_test'] >= error_conf)])

                full_hyrs_result['test_rejectsINC'] = sum(((self.data_model_dict['Ybtest'] != full_hyrs_preds)[
                    (self.data_model_dict['paccept_test'] >= mental_conf) & ~
                    self.data_model_dict['test_accept'] &
                    (self.data_model_dict[
                         'prob_human_wrong_test'] >= error_conf)]) &
                                                              (np.array(self.data_model_dict['Ybtest'] !=
                                                                        self.data_model_dict['Ytest'])[
                                                                  (self.data_model_dict[
                                                                       'paccept_test'] >= mental_conf) & ~
                                                                  self.data_model_dict['test_accept'] &
                                                                  (self.data_model_dict[
                                                                       'prob_human_wrong_test'] >= error_conf)]))

                full_hyrs_result['test_rejectsCOR'] = sum(((self.data_model_dict['Ybtest'] != full_hyrs_preds)[
                    (self.data_model_dict['paccept_test'] >= mental_conf) & ~
                    self.data_model_dict['test_accept'] &
                    (self.data_model_dict[
                         'prob_human_wrong_test'] >= error_conf)]) &
                                                              (np.array(self.data_model_dict['Ybtest'] ==
                                                                        self.data_model_dict['Ytest'])[
                                                                  (self.data_model_dict[
                                                                       'paccept_test'] >= mental_conf) & ~
                                                                  self.data_model_dict['test_accept'] &
                                                                  (self.data_model_dict[
                                                                       'prob_human_wrong_test'] >= error_conf)]))

                # allow human to reject preds
                full_hyrs_preds = deepcopy(self.hyrs_results['humanified_test_preds'])

                # reset instances where predicted acceptance is less than threshold to human response
                full_hyrs_preds[self.data_model_dict['paccept_test'] < mental_conf] = \
                    self.data_model_dict['Ybtest'][
                        self.data_model_dict['paccept_test'] < mental_conf]

                # reset instances where predicted chance of human error is greater than threshold to human response
                full_hyrs_preds[self.data_model_dict['prob_human_wrong_test'] < error_conf] = \
                    self.data_model_dict['Ybtest'][
                        self.data_model_dict['prob_human_wrong_test'] < error_conf]

                full_hyrs_covered = (pd.Series(self.hyrs_results['test_covered']) != -1) & \
                                        pd.Series(self.data_model_dict['paccept_test'] >= mental_conf) & \
                                        pd.Series(self.data_model_dict['prob_human_wrong_test'] >= error_conf)
                full_hyrs_result['test_coverage'] = (full_hyrs_covered == 1).sum()
                full_hyrs_result['test_error'] = 1 - metrics.accuracy_score(self.data_model_dict['Ytest'],
                                                                                full_hyrs_preds)
                full_hyrs_result['mental_conf'] = mental_conf
                full_hyrs_result['error_conf'] = error_conf
                full_hyrs_result['test_covereds'] = np.array(full_hyrs_covered)
                full_hyrs_result['humanified_test_preds'] = full_hyrs_preds


                full_hyrs_results.loc[index, :] = full_hyrs_result

                index += 1
        self.full_hyrs_results = full_hyrs_results

    def filter_tr_results(self, mental=False, error=False):
        #disregard filtering, it is related to other functionality we are working on and not to the paper, when set to false it should not filter

        if mental:
            mental_confs = [0, 0.0000001, 0.00001, 0.000025, 0.0001, 0.0005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.999, 0.9999,
                            0.99999,
                            0.999999, 0.9999999]
        else:
            mental_confs = [0]

        if error:
            error_confs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            error_confs = [0]

        full_tr_results = pd.DataFrame(index=range(len(mental_confs) * len(error_confs)),
                                           columns=['mental_conf',
                                                    'error_conf',
                                                    'test_coverage',
                                                    'test_rejects',
                                                    'test_rejectsINC',
                                                    'test_rejectsCOR',
                                                    'test_error',
                                                    'test_covereds',
                                                    'modelonly_test_preds',
                                                    'humanified_test_preds'])
        index = 0
        for mental_conf in mental_confs:
            for error_conf in error_confs:
                full_tr_result = {}
                full_tr_preds, _, _ = self.tr.predict(self.data_model_dict['Xtest'],
                                                              self.data_model_dict['Ybtest'])
                full_tr_result['test_rejects'] = sum((self.data_model_dict['Ybtest'] != full_tr_preds)[
                                                             (self.data_model_dict['paccept_test'] >= mental_conf) & ~
                                                             self.data_model_dict['test_accept'] &
                                                             (self.data_model_dict[
                                                                  'prob_human_wrong_test'] >= error_conf)])
                full_tr_result['modelonly_test_preds'] = self.tr_results['modelonly_test_preds']

                full_tr_result['test_rejectsINC'] = sum(((self.data_model_dict['Ybtest'] != full_tr_preds)[
                    (self.data_model_dict['paccept_test'] >= mental_conf) & ~
                    self.data_model_dict['test_accept'] &
                    (self.data_model_dict[
                         'prob_human_wrong_test'] >= error_conf)]) &
                                                            (np.array(
                                                                self.data_model_dict['Ybtest'] != self.data_model_dict[
                                                                    'Ytest'])[
                                                                (self.data_model_dict[
                                                                     'paccept_test'] >= mental_conf) & ~
                                                                self.data_model_dict['test_accept'] &
                                                                (self.data_model_dict[
                                                                     'prob_human_wrong_test'] >= error_conf)]))

                full_tr_result['test_rejectsCOR'] = sum(((self.data_model_dict['Ybtest'] != full_tr_preds)[
                    (self.data_model_dict['paccept_test'] >= mental_conf) & ~
                    self.data_model_dict['test_accept'] &
                    (self.data_model_dict[
                         'prob_human_wrong_test'] >= error_conf)]) &
                                                            (np.array(
                                                                self.data_model_dict['Ybtest'] == self.data_model_dict[
                                                                    'Ytest'])[
                                                                (self.data_model_dict[
                                                                     'paccept_test'] >= mental_conf) & ~
                                                                self.data_model_dict['test_accept'] &
                                                                (self.data_model_dict[
                                                                     'prob_human_wrong_test'] >= error_conf)]))

                # allow human to reject
                full_tr_preds = deepcopy(self.tr_results['humanified_test_preds'])

                # reset instances to human response where predicted mental is less than threshold
                full_tr_preds[self.data_model_dict['paccept_test'] < mental_conf] = self.data_model_dict['Ybtest'][
                    self.data_model_dict['paccept_test'] < mental_conf]

                # reset instances to human response where predicted chance of human error is less than threshold
                full_tr_preds[self.data_model_dict['prob_human_wrong_test'] < error_conf] = \
                    self.data_model_dict['Ybtest'][
                        self.data_model_dict['prob_human_wrong_test'] < error_conf]

                full_tr_covered = (pd.Series(self.tr_results['test_covered']) != -1) & pd.Series(
                    self.data_model_dict['paccept_test'] >= mental_conf) & \
                                      pd.Series(self.data_model_dict['prob_human_wrong_test'] >= error_conf)
                full_tr_result['test_coverage'] = (full_tr_covered == 1).sum()
                full_tr_result['test_coverage'] = (full_tr_covered == 1).sum()
                full_tr_result['test_error'] = 1 - metrics.accuracy_score(self.data_model_dict['Ytest'],
                                                                              full_tr_preds)
                full_tr_result['mental_conf'] = mental_conf
                full_tr_result['error_conf'] = error_conf
                full_tr_result['test_covereds'] = np.array(full_tr_covered)
                full_tr_result['humanified_test_preds'] = full_tr_preds

                full_tr_results.loc[index, :] = full_tr_result

                index += 1
        self.full_tr_results = full_tr_results

    def setup_brs(self):
        model = brs(self.data_model_dict['Xtrain'], self.data_model_dict['Ytrain'])
        model.generate_rules(self.supp,self.maxlen,self.Nrules, method='randomforest')
        model.set_parameters()
        self.brs_model = model

    def train_brs(self):
        self.brs_rules = self.brs_model.fit(self.iters, Nchain=1, print_message=False)
        modelonly_test_preds = brs_predict(self.brs_rules, self.data_model_dict['Xtest'])
        team_test_preds = modelonly_test_preds.copy()
        team_test_preds[self.data_model_dict['test_accept'] == False] = self.data_model_dict['Ybtest'][self.data_model_dict['test_accept'] == False]

        self.brs_results = pd.DataFrame({'test_error_brs':1 - metrics.accuracy_score(self.data_model_dict['Ytest'],team_test_preds),
                                         'test_error_modelonly':1 - metrics.accuracy_score(self.data_model_dict['Ytest'], modelonly_test_preds),
                                         'modelonly_test_preds':[modelonly_test_preds],
                                         'team_test_preds':[team_test_preds],

                                         'test_rejects':sum((modelonly_test_preds != self.data_model_dict['Ybtest']) &
                                                            (self.data_model_dict['test_accept']==False)),
                                         'test_rejectsCOR':sum((modelonly_test_preds != self.data_model_dict['Ybtest']) &
                                                               (self.data_model_dict['test_accept']==False) &
                                                               (self.data_model_dict['Ybtest'] == self.data_model_dict['Ytest']))})



    def setup_tr(self):
        model = tr(self.data_model_dict['Xtrain'], self.data_model_dict['Ytrain'],
                    self.data_model_dict['Ybtrain'],
                    self.data_model_dict['paccept_train'])

        model.set_parameters(self.alpha, self.beta, self.coverage_reg, self.contradiction_reg, self.fA, self.rejectType, self.force_complete_coverage, self.asym_loss, self.asym_accept)

        model.generate_rulespace(self.supp, self.maxlen, self.Nrules, need_negcode=True, method='randomforest',
                                 criteria='precision')
        

        self.tr = model

    def train_tr(self):
        iters = self.iters
        maps, accuracy_min, covered_min = self.tr.train(iters, T0=0.01, print_message=False)
        train_preds, train_covered, train_Yb = self.tr.predictHumanInLoop(self.data_model_dict['Xtrain'],
                                                                              self.data_model_dict['Ybtrain'],
                                                                              self.data_model_dict['train_accept'])

        modelonly_train_preds, _, _ = self.tr.predict(self.data_model_dict['Xtrain'],
                                                          self.data_model_dict['Ybtrain'])

        train_error_tr = 1 - metrics.accuracy_score(self.data_model_dict['Ytrain'],
                                                        train_preds)

        train_error_human = 1 - metrics.accuracy_score(self.data_model_dict['Ytrain'],
                                                       self.data_model_dict['Ybtrain'])

        modelonly_test_preds, _, _ = self.tr.predict(self.data_model_dict['Xtest'],
                                                         self.data_model_dict['Ybtest'])

        test_preds, test_covered, test_Yb = self.tr.predictHumanInLoop(self.data_model_dict['Xtest'],
                                                                           self.data_model_dict['Ybtest'],
                                                                           self.data_model_dict['test_accept'])

        soft_train_preds, soft_train_covered, soft_train_Yb = self.tr.predictSoft(self.data_model_dict['Xtrain'],
                                                                                      self.data_model_dict['Ybtrain'],
                                                                                      self.data_model_dict[
                                                                                          'paccept_train'])

        train_soft_error = 1 - metrics.accuracy_score(self.data_model_dict['Ytrain'],
                                                      soft_train_preds)

        test_error_tr = 1 - metrics.accuracy_score(self.data_model_dict['Ytest'],
                                                       test_preds)

        test_error_human = 1 - metrics.accuracy_score(self.data_model_dict['Ytest'],
                                                      self.data_model_dict['Ybtest'])

        soft_test_preds, soft_test_covered, soft_test_Yb = self.tr.predictSoft(self.data_model_dict['Xtest'],
                                                                                   self.data_model_dict['Ybtest'],
                                                                                   self.data_model_dict['paccept_test'])

        test_soft_error = 1 - metrics.accuracy_score(self.data_model_dict['Ytest'],
                                                     soft_test_preds)

        self.tr_results = {'train_error_tr': train_error_tr,
                               'train_error_human': train_error_human,
                               'train_soft_error': train_soft_error,
                               'train_rejects_soft': sum(self.data_model_dict['Ybtrain'][(train_covered != -1) & (
                                       self.data_model_dict['paccept_train'] <= 0.5)] != train_covered[
                                                             (train_covered != -1) & (self.data_model_dict[
                                                                                          'paccept_train'] <= 0.5)]),
                               'train_rejects_actual': sum(self.data_model_dict['Ybtrain'][(train_covered != -1) & (
                                   ~self.data_model_dict['train_accept'])] != train_covered[(train_covered != -1) & (
                                   ~self.data_model_dict['train_accept'])]),
                               'train_coverage': sum(train_covered != -1),
                               'train_covered': train_covered,
                               'test_error_tr': test_error_tr,
                               'test_error_human': test_error_human,
                               'test_soft_error': test_soft_error,
                               'soft_train_preds': soft_train_preds,
                               'test_rejects': sum(self.data_model_dict['Ybtest'][
                                                       (test_covered != -1) & (~self.data_model_dict['test_accept'])] !=
                                                   test_covered[
                                                       (test_covered != -1) & (~self.data_model_dict['test_accept'])]),
                               'test_rejectsINC': sum((self.data_model_dict['Ybtest'][
                                                           (test_covered != -1) & (
                                                               ~self.data_model_dict['test_accept'])] !=
                                                       test_covered[
                                                           (test_covered != -1) & (
                                                               ~self.data_model_dict['test_accept'])]) &
                                                      (self.data_model_dict['Ybtest'][
                                                           (test_covered != -1) & (
                                                               ~self.data_model_dict['test_accept'])] !=
                                                       np.array(self.data_model_dict['Ytest'])[
                                                           (test_covered != -1) & (
                                                               ~self.data_model_dict['test_accept'])])),

                               'test_rejectsCOR': sum((self.data_model_dict['Ybtest'][
                                                           (test_covered != -1) & (
                                                               ~self.data_model_dict['test_accept'])] !=
                                                       test_covered[
                                                           (test_covered != -1) & (
                                                               ~self.data_model_dict['test_accept'])]) &
                                                      (self.data_model_dict['Ybtest'][
                                                           (test_covered != -1) & (
                                                               ~self.data_model_dict['test_accept'])] ==
                                                       np.array(self.data_model_dict['Ytest'])[
                                                           (test_covered != -1) & (
                                                               ~self.data_model_dict['test_accept'])])),
                               'test_coverage': sum(test_covered != -1),
                               'modelonly_test_preds': modelonly_test_preds,
                               'modelonly_train_preds': modelonly_train_preds,
                               'test_covered': test_covered,
                               'humanified_test_preds': test_preds
                               }

    def setup_hyrs(self):
        model = hyrs.hyrs(self.data_model_dict['Xtrain'], self.data_model_dict['Ytrain'],
                                    self.data_model_dict['Ybtrain'])

        model.set_parameters(self.alpha, self.beta, self.coverage_reg, self.contradiction_reg, self.force_complete_coverage, self.asym_loss)
        model.generate_rulespace(self.supp, self.maxlen, self.Nrules, need_negcode=True, method='randomforest',
                                 criteria='precision')
        

        self.hyrs = model

    def train_hyrs(self):

        iters = self.iters
        maps, accuracy_min, covered_min = self.hyrs.train(iters, T0=0.01, print_message=False)

        train_preds, train_covered, train_Yb = self.hyrs.predict(self.data_model_dict['Xtrain'],
                                                                            self.data_model_dict['Ybtrain'])

        train_preds_collabing = self.hyrs.humanifyPreds(train_preds, self.data_model_dict['Ybtrain'],
                                                                   self.data_model_dict['train_accept'])

        train_error_hyrs = 1 - metrics.accuracy_score(self.data_model_dict['Ytrain'],
                                                          train_preds)

        train_error_collabing = 1 - metrics.accuracy_score(self.data_model_dict['Ytrain'],
                                                           train_preds_collabing)

        train_error_human = 1 - metrics.accuracy_score(self.data_model_dict['Ytrain'],
                                                       self.data_model_dict['Ybtrain'])

        test_preds, test_covered, test_Yb = self.hyrs.predict(self.data_model_dict['Xtest'],
                                                                         self.data_model_dict['Ybtest'])

        test_preds_collabing = self.hyrs.humanifyPreds(test_preds, self.data_model_dict['Ybtest'],
                                                                  self.data_model_dict['test_accept'])

        test_error_hyrs = 1 - metrics.accuracy_score(self.data_model_dict['Ytest'],
                                                         test_preds)

        test_error_collabing = 1 - metrics.accuracy_score(self.data_model_dict['Ytest'],
                                                          test_preds_collabing)

        test_error_human = 1 - metrics.accuracy_score(self.data_model_dict['Ytest'],
                                                      self.data_model_dict['Ybtest'])

        self.hyrs_results = {'train_error_hyrs': train_error_hyrs,
                                        'train_error_human': train_error_human,
                                        'train_rejects': sum(self.data_model_dict['Ybtrain'][
                                                                 (train_covered != -1) & (
                                                                     ~self.data_model_dict['train_accept'])] !=
                                                             train_covered[(train_covered != -1) & (
                                                                 ~self.data_model_dict['train_accept'])]),
                                        'train_coverage': sum(train_covered != -1),
                                        'train_error_collabing': train_error_collabing,
                                        'test_error_hyrs': test_error_hyrs,
                                        'test_error_collabing': test_error_collabing,
                                        'test_error_human': test_error_human,
                                        'test_rejects': sum(self.data_model_dict['Ybtest'][
                                                                (test_covered != -1) & (
                                                                    ~self.data_model_dict['test_accept'])] !=
                                                            test_covered[(test_covered != -1) & (
                                                                ~self.data_model_dict['test_accept'])]),
                                        'test_rejectsINC': sum((self.data_model_dict['Ybtest'][
                                                                    (test_covered != -1) & (
                                                                        ~self.data_model_dict['test_accept'])] !=
                                                                test_covered[(test_covered != -1) & (
                                                                    ~self.data_model_dict['test_accept'])]) &
                                                               (np.array(self.data_model_dict['Ytest'])[
                                                                    (test_covered != -1) & (
                                                                        ~self.data_model_dict['test_accept'])] ==
                                                                test_covered[(test_covered != -1) & (
                                                                    ~self.data_model_dict['test_accept'])])),

                                        'test_rejectsCOR': sum((self.data_model_dict['Ybtest'][
                                                                    (test_covered != -1) & (
                                                                        ~self.data_model_dict['test_accept'])] !=
                                                                test_covered[(test_covered != -1) & (
                                                                    ~self.data_model_dict['test_accept'])]) &
                                                               (np.array(self.data_model_dict['Ytest'])[
                                                                    (test_covered != -1) & (
                                                                        ~self.data_model_dict['test_accept'])] !=
                                                                test_covered[(test_covered != -1) & (
                                                                    ~self.data_model_dict['test_accept'])])),


                                        'test_coverage': sum(test_covered != -1),
                                        'modelonly_test_preds': test_preds,
                                        'modelyonly_train_preds': train_preds,
                                        'test_covered': test_covered,
                                        'train_covered': train_covered,
                                        'humanified_test_preds': test_preds_collabing,
                                        'humanified_train_preds': train_preds_collabing
                                        }