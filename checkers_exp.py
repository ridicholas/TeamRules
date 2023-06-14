from experiment_helper import *
import warnings

warnings.filterwarnings('ignore')
from util import *
import sklearn.metrics
from sklearn import metrics
from datetime import date
from copy import deepcopy
import pickle
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from runner import run



startDict = make_checkers()

accept_criteria=0.5

# make teams
team1 = HAI_team(startDict)

team3 = HAI_team(startDict)


# make humans
team1.make_human_model(type='logistic',
                       acceptThreshold=accept_criteria,
                       numExamplesToUse=500,
                       numColsToUse=0,
                       biasFactor=0,
                       alterations={'goodRange': [0, 0.5],
                                    'goodProb': 0.8,
                                    'badProb': 0.8,
                                    'badRange': [0.5, 1],
                                    'Rational': True,
                                    'adder': 0},
                       drop=['humangood'])

team3.make_human_model(type='logistic',
                       acceptThreshold=accept_criteria,
                       numExamplesToUse=500,
                       numColsToUse=0,
                       biasFactor=0,
                       alterations={'goodRange': [0, 0.5],
                                    'goodProb': 0.8,
                                    'badProb': 0.8,
                                    'badRange': [0.5, 1],
                                    'Rational': True,
                                    'adder': 0},
                       drop=['humangood'])


# make human good at their region and bad at other region
team1.makeHumanGood()
team3.makeHumanGood()


team2 = deepcopy(team1)

#adjust human confidences
team1.adjustConfidences(where_col='humangood', where_val=0, comp='negate')
team3.adjustConfidences(where_col='humangood', where_val=1)
team2.adjustConfidences(where_col=0, where_val=1)


team1.set_custom_confidence(team1.data_model_dict['train_conf'],
                            team1.data_model_dict['val_conf'],
                            team1.data_model_dict['test_conf'],
                            'deterministic')

team2.set_custom_confidence(team2.data_model_dict['train_conf'],
                            team2.data_model_dict['val_conf'],
                            team2.data_model_dict['test_conf'],
                            'deterministic')

team3.set_custom_confidence(team3.data_model_dict['train_conf'],
                            team3.data_model_dict['val_conf'],
                            team3.data_model_dict['test_conf'],
                            'deterministic')





teams = [team1, team2, team3]
for team in teams:
    team.train_humangood = team.data_model_dict['Xtrain']['humangood']
    team.val_humangood = team.data_model_dict['Xval']['humangood']
    team.test_humangood = team.data_model_dict['Xtest']['humangood']
    team.data_model_dict['Xtrain'].drop(columns=['humangood'], inplace=True)
    team.data_model_dict['Xval'].drop(columns=['humangood'], inplace=True)
    team.data_model_dict['Xtest'].drop(columns=['humangood'], inplace=True)
    team.post_human_dict['Xtrain'].drop(columns=['humangood'], inplace=True)
    team.post_human_dict['Xval'].drop(columns=['humangood'], inplace=True)
    team.post_human_dict['Xtest'].drop(columns=['humangood'], inplace=True)




for team in teams:
    team.post_human_dict = descretize_moons(team.data_model_dict)


team_info = pd.DataFrame(index=[1, 2, 3])

team1_2_start_threshold = 0.5
team3_4_start_threshold = 0.5

for i in range(1, 4):
    team_info.loc[i, 'Human Train Acc'] = metrics.accuracy_score(teams[i - 1].data_model_dict['Ytrain'],
                                                                 teams[i - 1].data_model_dict['Ybtrain'])

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

folder = 'checkers_contradiction_results'


team1_rule_lists = pd.DataFrame(index=range(0, 20), columns=[
                                'TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])
team2_rule_lists = pd.DataFrame(index=range(0, 20), columns=[
                                'TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])
team3_rule_lists = pd.DataFrame(index=range(0, 20), columns=[
                                'TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])


print('Starting Experiments....... \n')
run(team1, team2, team3, folder, team_info)

