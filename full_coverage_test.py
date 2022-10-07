from experiment_helper import *
import warnings
warnings.filterwarnings('ignore')
from util import *
import sklearn.metrics
from sklearn import metrics
from datetime import date
from copy import deepcopy
import pickle
import math
import random
from sklearn.preprocessing import MinMaxScaler

startDict = make_FICO_data(numQs=5)


#initial hyperparams
Niteration = 500
Nchain = 1
Nlevel = 1
Nrules = 10000
supp = 5
maxlen = 3
accept_criteria = 0.5
protected = 'NA'
budget = 1
sample_ratio = 1
alpha = 0
beta = 0
iters = 500
coverage_reg = 0
rejection_reg = 0
fA=0.5
force_complete_coverage = False
asym_loss = [1,1]
asym_accept = 0
rejectType = 'all'


#make teams
team1 = HAI_team(startDict)
team1.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio, alpha, beta, iters,coverage_reg, rejection_reg, fA, force_complete_coverage, asym_loss, asym_accept)

# make humans
team1.make_human_model(type='logistic',
                       acceptThreshold=accept_criteria,
                       numExamplesToUse=500,
                       numColsToUse=0,
                       biasFactor=0,
                       alterations={'goodRange': [0, 0.5],
                                    'goodProb': 1,
                                    'badProb': 0.5,
                                    'badRange': [0.5, 1],
                                    'Rational': True,
                                    'adder': 0.5})

team1.train_mental_aversion_model('perfect')
team1.train_mental_error_boundary_model()

contradiction_reg = 0
team1.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                  alpha, beta, iters, coverage_reg, contradiction_reg, fA, rejectType, force_complete_coverage, asym_loss, asym_accept)

team1.setup_tr()
team1.train_tr()

team1.setup_hyrs()
team1.train_hyrs()

print('done')

