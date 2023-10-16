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
from sklearn.metrics import confusion_matrix
import time
from runner import run
from scipy.stats import bernoulli, uniform

accept_criteria = 0.5





startDict = make_heart_data(numQs=5)


#make teams
team1 = HAI_team(startDict)
#team1.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio, alpha, beta, iters, fairness_reg, contradiction_reg, fA)
team3 = HAI_team(startDict)
#team3.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio, alpha, beta, iters, fairness_reg, contradiction_reg, fA)

# make humans
team1.make_human_model(type='logistic',
                       acceptThreshold=accept_criteria,
                       numExamplesToUse=500,
                       numColsToUse=0,
                       biasFactor=0,
                       alterations={'goodRange': [0, 0.3],
                                    'goodProb': 0.95,
                                    'badProb': 0.5,
                                    'badRange': [0.3, 1],
                                    'Rational': True,
                                    'adder': 0.6})

team3.make_human_model(type='logistic',
                       acceptThreshold=accept_criteria,
                       numExamplesToUse=500,
                       numColsToUse=0,
                       biasFactor=0,
                       alterations={'goodRange': [0, 0.3],
                                    'goodProb': 0.95,
                                    'badProb': 0.5,
                                    'badRange': [0.3, 1],
                                    'Rational': False,
                                    'adder': 0.6})


#good at young population, bad at old population
#team1.data_model_dict['Ybtrain'] = bernoulli.rvs(p=0.5, size=len(team1.data_model_dict['Ytrain']))
#team1.data_model_dict['Ybval'] = bernoulli.rvs(p=0.5, size=len(team1.data_model_dict['Yval']))
#team1.data_model_dict['Ybtest'] = bernoulli.rvs(p=0.5, size=len(team1.data_model_dict['Ytest']))

#team1.data_model_dict['Ybtrain'][np.where((team1.data_model_dict['Xtrain']['age54.0'] == 0))[0]] = team1.data_model_dict['Ytrain'][np.where((team1.data_model_dict['Xtrain']['age54.0'] == 0))[0]]
#team1.data_model_dict['Ybval'][np.where((team1.data_model_dict['Xval']['age54.0'] == 0))[0]] = np.array(team1.data_model_dict['Yval'])[np.where((team1.data_model_dict['Xval']['age54.0'] == 0))[0]]
#team1.data_model_dict['Ybtest'][np.where((team1.data_model_dict['Xtest']['age54.0'] == 0))[0]] = np.array(team1.data_model_dict['Ytest'])[np.where((team1.data_model_dict['Xtest']['age54.0'] == 0))[0]]



team2 = deepcopy(team1)

#good at males, bad at females
team2.data_model_dict['Ybtrain'] = bernoulli.rvs(p=0.5, size=len(team2.data_model_dict['Ytrain']))
team2.data_model_dict['Ybval'] = bernoulli.rvs(p=0.5, size=len(team2.data_model_dict['Yval']))
team2.data_model_dict['Ybtest'] = bernoulli.rvs(p=0.5, size=len(team2.data_model_dict['Ytest']))

team2.data_model_dict['Ybtrain'][np.where((team2.data_model_dict['Xtrain']['sex_Male'] == 1))[0]] = team2.data_model_dict['Ytrain'][np.where( (team2.data_model_dict['Xtrain']['sex_Male'] == 1))[0]]
team2.data_model_dict['Ybval'][np.where((team2.data_model_dict['Xval']['sex_Male'] == 1))[0]] = np.array(team2.data_model_dict['Yval'])[np.where((team2.data_model_dict['Xval']['sex_Male'] == 1))[0]]
team2.data_model_dict['Ybtest'][np.where( (team2.data_model_dict['Xtest']['sex_Male'] == 1))[0]] = np.array(team2.data_model_dict['Ytest'])[np.where((team2.data_model_dict['Xtest']['sex_Male'] == 1))[0]]

#lower "good accuracy" to 0.85
random_train = bernoulli.rvs(p=0.15, size=len(team2.data_model_dict['Ytrain']))
random_val = bernoulli.rvs(p=0.15, size=len(team2.data_model_dict['Yval']))
random_test = bernoulli.rvs(p=0.15, size=len(team2.data_model_dict['Ytest']))

team2.data_model_dict['Ybtrain'][np.where(((team2.data_model_dict['Xtrain']['sex_Male'] == 1) & (random_train==1)))[0]] = 1-team2.data_model_dict['Ybtrain'][np.where(((team2.data_model_dict['Xtrain']['sex_Male'] == 1) & (random_train==1)))[0]]
team2.data_model_dict['Ybval'][np.where(((team2.data_model_dict['Xval']['sex_Male'] == 1) & (random_val==1)))[0]] = 1-team2.data_model_dict['Ybval'][np.where(((team2.data_model_dict['Xval']['sex_Male'] == 1) & (random_val==1)))[0]]
team2.data_model_dict['Ybtest'][np.where(((team2.data_model_dict['Xtest']['sex_Male'] == 1) & (random_test==1)))[0]] = 1-team2.data_model_dict['Ybtest'][np.where(((team2.data_model_dict['Xtest']['sex_Male'] == 1) & (random_test==1)))[0]]                                                                     


#make neutral setting by making rational setting slightly less rational
train_conf2 = team2.data_model_dict['train_conf']
val_conf2 = team2.data_model_dict['val_conf']
test_conf2 = team2.data_model_dict['test_conf']


#train_conf1 = team1.data_model_dict['train_conf']
#val_conf1 = team1.data_model_dict['val_conf']
#test_conf1 = team1.data_model_dict['test_conf']

#confident at males, not confident at females
#train_conf2[np.where((team2.data_model_dict['Xtrain']['sex_Male'] == 1))] = np.random.randint(97,105,len(train_conf2[np.where((team2.data_model_dict['Xtrain']['sex_Male'] == 1))]))/100
#train_conf2[np.where((team2.data_model_dict['Xtrain']['sex_Male'] == 0))] = np.random.normal(30,10,len(train_conf2[np.where((team2.data_model_dict['Xtrain']['sex_Male'] == 0))]))/100
#val_conf2[np.where( (team2.data_model_dict['Xval']['sex_Male'] == 1))] = np.random.randint(97,105,len(val_conf2[np.where( (team2.data_model_dict['Xval']['sex_Male'] == 1))]))/100
#val_conf2[np.where((team2.data_model_dict['Xval']['sex_Male'] == 0))] = np.random.normal(30,10,len(val_conf2[np.where((team2.data_model_dict['Xval']['sex_Male'] == 0))]))/100

#test_conf2[np.where((team2.data_model_dict['Xtest']['sex_Male'] == 1))] = np.random.randint(97,105,len(test_conf2[np.where((team2.data_model_dict['Xtest']['sex_Male'] == 1))]))/100
#test_conf2[np.where( (team2.data_model_dict['Xtest']['sex_Male'] == 0))] = np.random.normal(30,10,len(test_conf2[np.where((team2.data_model_dict['Xtest']['sex_Male'] == 0))]))/100

#confident at young, not confident at elderly
train_conf2[np.where((team1.data_model_dict['Xtrain']['age54.0'] == 0))] = np.random.randint(97,105,len(train_conf2[np.where((team1.data_model_dict['Xtrain']['age54.0'] == 0) )]))/100
train_conf2[np.where((team1.data_model_dict['Xtrain']['age54.0'] == 1))] = np.random.normal(30,10,len(train_conf2[np.where((team1.data_model_dict['Xtrain']['age54.0'] == 1) )]))/100
val_conf2[np.where((team1.data_model_dict['Xval']['age54.0'] == 0) )] = np.random.randint(97,105,len(val_conf2[np.where((team1.data_model_dict['Xval']['age54.0'] == 0) )]))/100
val_conf2[np.where((team1.data_model_dict['Xval']['age54.0'] == 1) )] = np.random.normal(30,10,len(val_conf2[np.where((team1.data_model_dict['Xval']['age54.0'] == 1) )]))/100

test_conf2[np.where((team1.data_model_dict['Xtest']['age54.0'] == 0) )] = np.random.randint(97,105,len(test_conf2[np.where((team1.data_model_dict['Xtest']['age54.0'] == 0) )]))/100
test_conf2[np.where((team1.data_model_dict['Xtest']['age54.0'] == 1) )] = np.random.normal(30,10,len(test_conf2[np.where((team1.data_model_dict['Xtest']['age54.0'] == 1) )]))/100







team2.set_custom_confidence(team2.data_model_dict['train_conf'],
                            team2.data_model_dict['val_conf'],
                            team2.data_model_dict['test_conf'],
                            'deterministic')



#team1.set_custom_confidence(team1.data_model_dict['train_conf'],
#                            team1.data_model_dict['val_conf'],
#                            team1.data_model_dict['test_conf'],
#                            'deterministic')


teams = [team1, team2, team3]


team_info = pd.DataFrame(index=[1, 2, 3])

team1_2_start_threshold = 0.5
team3_4_start_threshold = 0.5

for i in range(1, 4):
    team_info.loc[i,'Human Train Acc'] = metrics.accuracy_score(teams[i-1].data_model_dict['Ytrain'], teams[i-1].data_model_dict['Ybtrain'])

team1.accept_criteria = team1_2_start_threshold
team2.accept_criteria = team1_2_start_threshold

team_info.loc[1, 'accept_threshold'] = team1_2_start_threshold
team_info.loc[2, 'accept_threshold'] = team1_2_start_threshold



team_info.loc[1, 'human true train accepts'] = (team1.data_model_dict['train_accept']).sum()
team_info.loc[1, 'human true train rejects'] = (~team1.data_model_dict['train_accept']).sum()
team_info.loc[2, 'human true train accepts'] = (team2.data_model_dict['train_accept']).sum()
team_info.loc[2, 'human true train rejects'] = (~team2.data_model_dict['train_accept']).sum()


print('human accuracy in accept region: {}'.format(metrics.accuracy_score(team1.data_model_dict['Ybtrain'][team1.data_model_dict['train_accept']],
                                                                          team1.data_model_dict['Ytrain'][team1.data_model_dict['train_accept']])))
print('human accuracy in reject region: {}'.format(metrics.accuracy_score(team1.data_model_dict['Ybtrain'][~team1.data_model_dict['train_accept']],
                                                                          team1.data_model_dict['Ytrain'][~team1.data_model_dict['train_accept']])))

team_info.loc[1, 'human accept region train acc'] = metrics.accuracy_score(team1.data_model_dict['Ybtrain'][team1.data_model_dict['train_accept']],
                                                                           team1.data_model_dict['Ytrain'][team1.data_model_dict['train_accept']])
team_info.loc[1, 'human reject region train acc'] = metrics.accuracy_score(team1.data_model_dict['Ybtrain'][~team1.data_model_dict['train_accept']],
                                                                           team1.data_model_dict['Ytrain'][~team1.data_model_dict['train_accept']])
team_info.loc[2, 'human accept region train acc'] = metrics.accuracy_score(team2.data_model_dict['Ybtrain'][team2.data_model_dict['train_accept']],
                                                                           team2.data_model_dict['Ytrain'][team2.data_model_dict['train_accept']])
team_info.loc[2, 'human reject region train acc'] = metrics.accuracy_score(team2.data_model_dict['Ybtrain'][~team2.data_model_dict['train_accept']],
                                                                           team2.data_model_dict['Ytrain'][~team2.data_model_dict['train_accept']])

team3.accept_criteria = team3_4_start_threshold

team_info.loc[3, 'accept_threshold'] = team3_4_start_threshold



team_info.loc[3, 'human true train accepts'] = (team3.data_model_dict['train_accept']).sum()
team_info.loc[3, 'human true train rejects'] = (~team3.data_model_dict['train_accept']).sum()

print('human accuracy in accept region: {}'.format(metrics.accuracy_score(team3.data_model_dict['Ybtrain'][team3.data_model_dict['train_accept']],
                                                                          team3.data_model_dict['Ytrain'][team3.data_model_dict['train_accept']])))
print('human accuracy in reject region: {}'.format(metrics.accuracy_score(team3.data_model_dict['Ybtrain'][~team3.data_model_dict['train_accept']],
                                                                          team3.data_model_dict['Ytrain'][~team3.data_model_dict['train_accept']])))

team_info.loc[3, 'human accept region train acc'] = metrics.accuracy_score(team3.data_model_dict['Ybtrain'][team3.data_model_dict['train_accept']],
                                                                           team3.data_model_dict['Ytrain'][team3.data_model_dict['train_accept']])
team_info.loc[3, 'human reject region train acc'] = metrics.accuracy_score(team3.data_model_dict['Ybtrain'][~team3.data_model_dict['train_accept']],
                                                                           team3.data_model_dict['Ytrain'][~team3.data_model_dict['train_accept']])

print(team_info)

folder = '1_heart_contradiction_results'


team1_rule_lists = pd.DataFrame(index=range(0, 20), columns=['TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])
team2_rule_lists = pd.DataFrame(index=range(0, 20), columns=['TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])
team3_rule_lists = pd.DataFrame(index=range(0, 20), columns=['TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])





print('Starting Experiments....... \n')
run(team1, team2, team3, folder, team_info)



