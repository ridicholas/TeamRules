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
from runner import run

startDict = make_Adult_data()


accept_criteria = 0.5




# make teams
team1 = HAI_team(startDict)

team3 = HAI_team(startDict)


# make humans
team1.make_human_model(type='logistic',
                       acceptThreshold=accept_criteria,
                       numExamplesToUse=500,
                       numColsToUse=0,
                       biasFactor=0,
                       alterations={'goodRange': [0, 0.4],
                                    'goodProb': 1,
                                    'badProb': 0.5,
                                    'badRange': [0.5, 0.8],
                                    'Rational': True,
                                    'adder': 0.4})

team3.make_human_model(type='logistic',
                       acceptThreshold=accept_criteria,
                       numExamplesToUse=500,
                       numColsToUse=0,
                       biasFactor=0,
                       alterations={'goodRange': [0, 0.4],
                                    'goodProb': 1,
                                    'badProb': 0.5,
                                    'badRange': [0.5, 0.8],
                                    'Rational': False,
                                    'adder': 0.4})






team2 = deepcopy(team1)


train_conf2 = team2.data_model_dict['train_conf']
val_conf2 = team2.data_model_dict['val_conf']
test_conf2 = team2.data_model_dict['test_conf']

train_conf2[np.where(team2.data_model_dict['Xtrain']['occupation_Exec-managerial'] == 0)] = 0.8
train_conf2[np.where(team2.data_model_dict['Xtrain']['occupation_Exec-managerial'] == 1)] = 0.2
val_conf2[np.where(team2.data_model_dict['Xval']['occupation_Exec-managerial'] == 0)] = 0.8
val_conf2[np.where(team2.data_model_dict['Xval']['occupation_Exec-managerial'] == 1)] = 0.2

test_conf2[np.where(team2.data_model_dict['Xtest']['occupation_Exec-managerial'] == 0)] = 0.8
test_conf2[np.where(team2.data_model_dict['Xtest']['occupation_Exec-managerial'] == 1)] = 0.2



team2.set_custom_confidence(train_conf2, val_conf2, test_conf2, 'deterministic')


teams = [team1, team2, team3]

team_info = pd.DataFrame(index=[1, 2, 3])

team1_2_start_threshold = 0.5
team3_4_start_threshold = 0.5

for i in range(1, 4):
    team_info.loc[i, 'Human Train Acc'] = metrics.accuracy_score(teams[i-1].data_model_dict['Ytrain'], teams[i-1].data_model_dict['Ybtrain'])


team_info.loc[1, 'human true train accepts'] = (team1.data_model_dict['train_accept']).sum()
team_info.loc[1, 'human true train rejects'] = (~team1.data_model_dict['train_accept']).sum()
team_info.loc[2, 'human true train accepts'] = (team2.data_model_dict['train_accept']).sum()
team_info.loc[2, 'human true train rejects'] = (~team2.data_model_dict['train_accept']).sum()

print('human accuracy in accept region: {}'.format(
    metrics.accuracy_score(team1.data_model_dict['Ybtrain'][team1.data_model_dict['train_accept']],
                           team1.data_model_dict['Ytrain'][team1.data_model_dict['train_accept']])))
print('human accuracy in reject region: {}'.format(
    metrics.accuracy_score(team1.data_model_dict['Ybtrain'][~team1.data_model_dict['train_accept']],
                           team1.data_model_dict['Ytrain'][~team1.data_model_dict['train_accept']])))

team_info.loc[1, 'human accept region train acc'] = metrics.accuracy_score(
    team1.data_model_dict['Ybtrain'][team1.data_model_dict['train_accept']],
    team1.data_model_dict['Ytrain'][team1.data_model_dict['train_accept']])
team_info.loc[1, 'human reject region train acc'] = metrics.accuracy_score(
    team1.data_model_dict['Ybtrain'][~team1.data_model_dict['train_accept']],
    team1.data_model_dict['Ytrain'][~team1.data_model_dict['train_accept']])
team_info.loc[2, 'human accept region train acc'] = metrics.accuracy_score(
    team2.data_model_dict['Ybtrain'][team2.data_model_dict['train_accept']],
    team2.data_model_dict['Ytrain'][team2.data_model_dict['train_accept']])
team_info.loc[2, 'human reject region train acc'] = metrics.accuracy_score(
    team2.data_model_dict['Ybtrain'][~team2.data_model_dict['train_accept']],
    team2.data_model_dict['Ytrain'][~team2.data_model_dict['train_accept']])


team_info.loc[3, 'human true train accepts'] = (team3.data_model_dict['train_accept']).sum()
team_info.loc[3, 'human true train rejects'] = (~team3.data_model_dict['train_accept']).sum()

print('human accuracy in accept region: {}'.format(
    metrics.accuracy_score(team3.data_model_dict['Ybtrain'][team3.data_model_dict['train_accept']],
                           team3.data_model_dict['Ytrain'][team3.data_model_dict['train_accept']])))
print('human accuracy in reject region: {}'.format(
    metrics.accuracy_score(team3.data_model_dict['Ybtrain'][~team3.data_model_dict['train_accept']],
                           team3.data_model_dict['Ytrain'][~team3.data_model_dict['train_accept']])))

team_info.loc[3, 'human accept region train acc'] = metrics.accuracy_score(
    team3.data_model_dict['Ybtrain'][team3.data_model_dict['train_accept']],
    team3.data_model_dict['Ytrain'][team3.data_model_dict['train_accept']])
team_info.loc[3, 'human reject region train acc'] = metrics.accuracy_score(
    team3.data_model_dict['Ybtrain'][~team3.data_model_dict['train_accept']],
    team3.data_model_dict['Ytrain'][~team3.data_model_dict['train_accept']])

print(team_info)

folder = 'adult_contradiction_results'


team1_rule_lists = pd.DataFrame(index=range(0, 20), columns=['TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])
team2_rule_lists = pd.DataFrame(index=range(0, 20), columns=['TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])
team3_rule_lists = pd.DataFrame(index=range(0, 20), columns=['TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])

print('Starting Experiments....... \n')
run(team1, team2, team3, folder, team_info)