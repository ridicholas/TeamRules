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
iters = Niteration
coverage_reg = 0
rejection_reg = 0


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
team1.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio, alpha, beta, iters,coverage_reg, rejection_reg, fA)
team3 = HAI_team(startDict)
team3.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio, alpha, beta, iters,coverage_reg, rejection_reg, fA)

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


team2.set_custom_confidence(train_conf2, val_conf2, test_conf2, 'deterministic')

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

print('Starting Experiments....... \n')
# Repeat Experiments
for run in range(0, 10):

    team_info = pd.DataFrame(index=[1, 2, 3])
    coverage_regs = [0, 0.01, 0.05, 0.1, 0.2,
                     0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1]

    coverage_reg = 0
    contradiction_reg = 0 
    # split training and test randomly
    team1.makeAdditionalTestSplit(testPercent=0.2, replaceExisting=True, random_state=run, others=[team2, team3])

    startData = pd.DataFrame({'team1_Ytest': team1.data_model_dict['Ytest'],
                              'team1_Ybtest': team1.data_model_dict['Ybtest'],
                              'team2_Ytest': team2.data_model_dict['Ytest'],
                              'team2_Ybtest': team2.data_model_dict['Ybtest'],
                              'team3_Ytest': team3.data_model_dict['Ytest'],
                              'team3_Ybtest': team3.data_model_dict['Ybtest'],
                              'team1TestConf': team1.data_model_dict['test_conf'],
                              'team2TestConf': team2.data_model_dict['test_conf'],
                              'team3TestConf': team3.data_model_dict['test_conf'],
                              'testDex': team1.testDex})

    startData.to_pickle('{}/dataset_run{}.pkl'.format(folder, run))

    for i in range(1, 4):
        team_info.loc[i, 'Human Test Acc'] = metrics.accuracy_score(teams[i - 1].data_model_dict['Ytest'],
                                                                    teams[i - 1].data_model_dict['Ybtest'])

    # train aversion and error boundary models
    team1.train_mental_aversion_model('perfect')
    team1.train_confidence_model('perfect')
    team1.train_mental_error_boundary_model()
    team_info.loc[1, 'human true accepts'] = (team1.data_model_dict['test_conf'] < team1_2_start_threshold).sum()
    team_info.loc[1, 'human true rejects'] = (team1.data_model_dict['test_conf'] >= team1_2_start_threshold).sum()
    team_info.loc[1, 'human accept region test acc'] = metrics.accuracy_score(
        team1.data_model_dict['Ybtest'][team1.data_model_dict['test_accept']],
        team1.data_model_dict['Ytest'][team1.data_model_dict['test_accept']])
    team_info.loc[1, 'human reject region test acc'] = metrics.accuracy_score(
        team1.data_model_dict['Ybtest'][~team1.data_model_dict['test_accept']],
        team1.data_model_dict['Ytest'][~team1.data_model_dict['test_accept']])
    team_info.loc[1, 'aversion model test acc'] = metrics.accuracy_score(team1.data_model_dict['paccept_test'] > 0.5,
                                                                         team1.data_model_dict['test_accept'])

    team2.train_mental_aversion_model('perfect')
    team2.train_confidence_model('perfect')
    team2.train_mental_error_boundary_model()
    team_info.loc[2, 'human true accepts'] = (team2.data_model_dict['test_conf'] < team1_2_start_threshold).sum()
    team_info.loc[2, 'human true rejects'] = (team2.data_model_dict['test_conf'] >= team1_2_start_threshold).sum()
    team_info.loc[2, 'human accept region test acc'] = metrics.accuracy_score(
        team2.data_model_dict['Ybtest'][team2.data_model_dict['test_accept']],
        team2.data_model_dict['Ytest'][team2.data_model_dict['test_accept']])
    team_info.loc[2, 'human reject region test acc'] = metrics.accuracy_score(
        team2.data_model_dict['Ybtest'][~team2.data_model_dict['test_accept']],
        team2.data_model_dict['Ytest'][~team2.data_model_dict['test_accept']])
    team_info.loc[2, 'aversion model test acc'] = metrics.accuracy_score(team2.data_model_dict['paccept_test'] > 0.5,
                                                                         team2.data_model_dict['test_accept'])

    team3.train_mental_aversion_model('perfect')
    team3.train_confidence_model('perfect')
    team3.train_mental_error_boundary_model()
    team_info.loc[3, 'human true accepts'] = (team3.data_model_dict['test_conf'] < team3_4_start_threshold).sum()
    team_info.loc[3, 'human true rejects'] = (team3.data_model_dict['test_conf'] >= team3_4_start_threshold).sum()
    team_info.loc[3, 'human accept region test acc'] = metrics.accuracy_score(
        team3.data_model_dict['Ybtest'][team3.data_model_dict['test_accept']],
        team3.data_model_dict['Ytest'][team3.data_model_dict['test_accept']])
    team_info.loc[3, 'human reject region test acc'] = metrics.accuracy_score(
        team3.data_model_dict['Ybtest'][~team3.data_model_dict['test_accept']],
        team3.data_model_dict['Ytest'][~team3.data_model_dict['test_accept']])
    team_info.loc[3, 'aversion model test acc'] = metrics.accuracy_score(team3.data_model_dict['paccept_test'] > 0.5,
                                                                         team3.data_model_dict['test_accept'])

    team_info.to_pickle('{}/team_info_run{}.pkl'.format(folder, run))

    for reg in coverage_regs:
        contradiction_reg = reg

        
        print('training team1 hyrs model...')
        # train hyrs baseline
        
        team1.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                  alpha,
                                  beta, iters, coverage_reg, contradiction_reg, fA)
        team1.setup_hyrs()
        team1.train_hyrs()
        team1.filter_hyrs_results(mental=True, error=False)

        
        print('training team1 tr model...')
        team1.setup_tr()
        team1.train_tr()
        team1.filter_tr_results(mental=True, error=False)
        

        '''
        if contradiction_reg == 0:
            print('training team1 brs model...')
            team1.setup_brs()
            team1.train_brs()
            # print(team1.brs_results['test_error_modelonly'])
        '''

        
        print('training team2 hyrs model...')
        team2.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                  alpha,
                                  beta, iters, coverage_reg, contradiction_reg, fA)
        
        team2.setup_hyrs()
        team2.train_hyrs()
        team2.filter_hyrs_results(mental=True, error=False)
        
        print('training team2 tr model...')
        team2.setup_tr()
        team2.train_tr()
        team2.filter_tr_results(mental=True, error=False)
        
        '''
        if contradiction_reg == 0:
            print('training team2 brs model...')
            team2.setup_brs()
            team2.train_brs()
            # print(team2.brs_results['test_error_modelonly'])
        '''

        print('training team3 hyrs model...')
        team3.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                  alpha,
                                  beta, iters, coverage_reg, contradiction_reg, fA)
        team3.setup_hyrs()
        team3.train_hyrs()
        team3.filter_hyrs_results(mental=True, error=False)

        print('training team3 tr model...')
        team3.setup_tr()
        team3.train_tr()
        team3.filter_tr_results(mental=True, error=False)
        
        '''
        if contradiction_reg == 0:
            print('training team3 brs model...')
            team3.setup_brs()
            team3.train_brs()
            #print(team3.brs_results['test_error_modelonly'])
        '''
        
        
        # append rules lists
        team1_rule_lists.loc[run, 'TR_prules'] = team1.tr.prs_min
        team1_rule_lists.loc[run, 'TR_nrules'] = team1.tr.nrs_min
        team1_rule_lists.loc[run, 'HyRS_prules'] = team1.hyrs.prs_min
        team1_rule_lists.loc[run, 'HyRS_nrules'] = team1.hyrs.nrs_min

        team2_rule_lists.loc[run, 'TR_prules'] = team2.tr.prs_min
        team2_rule_lists.loc[run, 'TR_nrules'] = team2.tr.nrs_min
        team2_rule_lists.loc[run, 'HyRS_prules'] = team2.hyrs.prs_min
        team2_rule_lists.loc[run, 'HyRS_nrules'] = team2.hyrs.nrs_min

        team3_rule_lists.loc[run, 'TR_prules'] = team3.tr.prs_min
        team3_rule_lists.loc[run, 'TR_nrules'] = team3.tr.nrs_min
        team3_rule_lists.loc[run, 'HyRS_prules'] = team3.hyrs.prs_min
        team3_rule_lists.loc[run, 'HyRS_nrules'] = team3.hyrs.nrs_min

        # write results
        print('writing results...')
        
        # team1
        
        
        team1.full_hyrs_results.to_pickle('{}/cost_{}_team1_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
        
        team1.full_tr_results.to_pickle('{}/cost_{}_team1_tr_filtered_run{}.pkl'.format(folder, reg, run))

        with open('{}/cost_{}_team1_tr_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
            pickle.dump(team1.tr_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()
        
        with open('{}/cost_{}_team1_hyrs_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
            pickle.dump(team1.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()
        
        
        # team2
        team2.full_hyrs_results.to_pickle('{}/cost_{}_team2_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
        team2.full_tr_results.to_pickle('{}/cost_{}_team2_tr_filtered_run{}.pkl'.format(folder, reg, run))

        with open('{}/cost_{}_team2_tr_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
            pickle.dump(team2.tr_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()

        with open('{}/cost_{}_team2_hyrs_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
            pickle.dump(team2.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()
        
        # team3

        team3.full_hyrs_results.to_pickle('{}/cost_{}_team3_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
        team3.full_tr_results.to_pickle('{}/cost_{}_team3_tr_filtered_run{}.pkl'.format(folder, reg, run))

        with open('{}/cost_{}_team3_tr_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
            pickle.dump(team3.tr_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()

        with open('{}/cost_{}_team3_hyrs_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
            pickle.dump(team3.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()
        
        '''
        if contradiction_reg == 0:
            team1.brs_results.to_pickle('{}/team1_brs_run{}.pkl'.format(folder, run))
            team2.brs_results.to_pickle('{}/team2_brs_run{}.pkl'.format(folder, run))
            team3.brs_results.to_pickle('{}/team3_brs_run{}.pkl'.format(folder, run))
        
        
        #forced coverage versions
        
        team1.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                  alpha,
                                  beta, iters, coverage_reg, contradiction_reg, fA, force_complete_coverage=True)
        team1.setup_hyrs()
        team1.train_hyrs()
        team1.filter_hyrs_results(mental=True, error=False)


        print('training team1 tr model...')
        team1.setup_tr()
        team1.train_tr()
        team1.filter_tr_results(mental=True, error=False)

        team2.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                  alpha,
                                  beta, iters, coverage_reg, contradiction_reg, fA, force_complete_coverage=True)
        team2.setup_hyrs()
        team2.train_hyrs()
        team2.filter_hyrs_results(mental=True, error=False)


        print('training team2 tr model...')
        team2.setup_tr()
        team2.train_tr()
        team2.filter_tr_results(mental=True, error=False)

        team3.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                  alpha,
                                  beta, iters, coverage_reg, contradiction_reg, fA, force_complete_coverage=True)
        team3.setup_hyrs()
        team3.train_hyrs()
        team3.filter_hyrs_results(mental=True, error=False)


        print('training team3 tr model...')
        team3.setup_tr()
        team3.train_tr()
        team3.filter_tr_results(mental=True, error=False)

        # team1
        
        team1.full_hyrs_results.to_pickle('{}/fc_cost_{}_team1_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
        team1.full_tr_results.to_pickle('{}/fc_cost_{}_team1_tr_filtered_run{}.pkl'.format(folder, reg, run))

        with open('{}/fc_cost_{}_team1_tr_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
            pickle.dump(team1.tr_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()

        with open('{}/fc_cost_{}_team1_hyrs_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
            pickle.dump(team1.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()

        # team2
        team2.full_hyrs_results.to_pickle('{}/fc_cost_{}_team2_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
        team2.full_tr_results.to_pickle('{}/fc_cost_{}_team2_tr_filtered_run{}.pkl'.format(folder, reg, run))

        with open('{}/fc_cost_{}_team2_tr_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
            pickle.dump(team2.tr_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()

        with open('{}/fc_cost_{}_team2_hyrs_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
            pickle.dump(team2.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()

        # team3

        team3.full_hyrs_results.to_pickle('{}/fc_cost_{}_team3_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
        team3.full_tr_results.to_pickle('{}/fc_cost_{}_team3_tr_filtered_run{}.pkl'.format(folder, reg, run))

        with open('{}/fc_cost_{}_team3_tr_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
            pickle.dump(team3.tr_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()

        with open('{}/fc_cost_{}_team3_hyrs_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
            pickle.dump(team3.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()
        '''

        
print('finally done.')

