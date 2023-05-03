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

startDict = make_FICO_data(numQs=5)


#initial hyperparams
Niteration = 1000
Nchain = 1
Nlevel = 1
Nrules = 10000
supp = 5
maxlen = 7
accept_criteria = 0.5
protected = 'NA'
budget = 1
sample_ratio = 1
alpha = 0
beta = 0
iters = Niteration
contradiction_reg = 0
fairness_reg = 0


def basic_ADB_func_det(c_human, c_model=None, agreement=None):
    #returns the probability that the human accepts a recommendation given conf of human and conf of model
    return c_human <= 0.5

def basic_ADB_func_prob(c_human, c_model=None, agreement=None):
    #returns the probability that the human accepts a recommendation given conf of human and conf of model
    return 1-c_human

def basic_ADB_predicted_accept(paccept):
    #returns back the passed in probability that the human accepts a recommendation given feature values
    return paccept

def complex_ADB(c_human, c_model, agreement, delta=5, beta=0.05, k=0.63, gamma=0.95):
    #from will you accept the AI recommendation
    def w(p, k):
        return (p**k)/((p**k)+(1-p)**k)
    

    
    c_human_new = c_human.copy()
    c_human_new[c_human_new <= 0] = 0.0000001
    c_human_new[c_human_new >= 1] = 0.9999999

    if c_model.min() >= 0.5:
        scaler = MinMaxScaler((0,1))
        scaler.fit(np.array(c_model).reshape(-1,1))
        c_model_new = scaler.transform(np.array(c_model).reshape(-1,1)).flatten()
    else:
        c_model_new = c_model.copy()
    c_model_new[c_model_new <= 0] = 0.0000001
    c_model_new[c_model_new >= 1] = 0.9999999
    
    c_human_new[~agreement] = 1-c_human_new[~agreement]
    a = (c_model_new**gamma)/((c_model_new**gamma)+((1-c_model_new)**gamma))
    b = (c_human_new**gamma)/((c_human_new**gamma)+((1-c_human_new)**gamma))
    
    conf = 1/(1+(((1-a)*(1-b))/(a*b)))
    
    util_accept = (1+beta)*w(conf,k)-beta
    util_reject = 1-(1+beta)*w(conf,k)


    prob = np.exp(delta*util_accept)/(np.exp(delta*util_accept)+np.exp(delta*util_reject))
    df = pd.DataFrame({'c_human': c_human, 'c_human_new' : c_human_new, 'conf' : conf, 'c_model':c_model, 'agreement':agreement, 'prob': prob})

    return prob


fA=complex_ADB


#make teams
team1 = HAI_team(startDict)
team1.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio, alpha, beta, iters, fairness_reg, contradiction_reg, fA)
team3 = HAI_team(startDict)
team3.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio, alpha, beta, iters, fairness_reg, contradiction_reg, fA)

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

team3.make_human_model(type='logistic',
                       acceptThreshold=accept_criteria,
                       numExamplesToUse=500,
                       numColsToUse=0,
                       biasFactor=0,
                       alterations={'goodRange': [0, 0.5],
                                    'goodProb': 1,
                                    'badProb': 0.5,
                                    'badRange': [0.5, 1],
                                    'Rational': False,
                                    'adder': 0.5})



team2 = deepcopy(team1)


#make neutral setting by making rational setting slightly less rational
train_conf2 = team2.data_model_dict['train_conf']
val_conf2 = team2.data_model_dict['val_conf']
test_conf2 = team2.data_model_dict['test_conf']

train_conf2[np.where((team2.data_model_dict['Xtrain']['ExternalRiskEstimate65.0'] == 0) | (team2.data_model_dict['Xtrain']['NumSatisfactoryTrades24.0'] == 0))] = 0.8
train_conf2[np.where((team2.data_model_dict['Xtrain']['ExternalRiskEstimate65.0'] == 1) & (team2.data_model_dict['Xtrain']['NumSatisfactoryTrades24.0'] == 1))] = 0.2
val_conf2[np.where((team2.data_model_dict['Xval']['ExternalRiskEstimate65.0'] == 0) | (team2.data_model_dict['Xval']['NumSatisfactoryTrades24.0'] == 0))] = 0.8
val_conf2[np.where((team2.data_model_dict['Xval']['ExternalRiskEstimate65.0'] == 1) & (team2.data_model_dict['Xval']['NumSatisfactoryTrades24.0'] == 1))] = 0.2

test_conf2[np.where((team2.data_model_dict['Xtest']['ExternalRiskEstimate65.0'] == 0) | (team2.data_model_dict['Xtest']['NumSatisfactoryTrades24.0'] == 0))] = 0.8
test_conf2[np.where((team2.data_model_dict['Xtest']['ExternalRiskEstimate65.0'] == 1) & (team2.data_model_dict['Xtest']['NumSatisfactoryTrades24.0'] == 1))] = 0.2







team2.set_custom_confidence(team2.data_model_dict['train_conf'],
                            team2.data_model_dict['val_conf'],
                            team2.data_model_dict['test_conf'],
                            'deterministic')






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

folder = 'fico_contradiction_results'
team_info.to_pickle('{}/start_info.pkl'.format(folder))

team1.data_model_dict['Xtrain'].to_pickle('{}/startDataSet.pkl'.format(folder))

team1_rule_lists = pd.DataFrame(index=range(0, 20), columns=['TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])
team2_rule_lists = pd.DataFrame(index=range(0, 20), columns=['TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])
team3_rule_lists = pd.DataFrame(index=range(0, 20), columns=['TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])



def my_loss(contradiction_reg, ):
    def custom_loss(y_pred, y_true):
        a,g = alpha, gamma
        def fl(x,t):
            p = 1/(1+np.exp(-x))
            return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
        partial_fl = lambda x: fl(x, y_true)
        grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
        hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
        return grad, hess
    return custom_loss

xgb = xgb.XGBClassifier(objective=focal_loss(alpha=0.25, gamma=1))