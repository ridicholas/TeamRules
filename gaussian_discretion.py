from sklearn import tree
import pickle
from copy import deepcopy
from datetime import date
from sklearn import metrics
import sklearn.metrics
from util import *
from hmac import trans_36
from experiment_helper import *
import warnings

warnings.filterwarnings('ignore')


startDict = make_gaussians()

# initial hyperparams
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
contradiction_reg = 0
fA = 0.5

# make teams
team1 = HAI_team(startDict)
team1.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio, alpha,
                          beta, iters, coverage_reg, contradiction_reg, fA)


team3 = HAI_team(startDict)
team3.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio, alpha,
                          beta, iters, coverage_reg, contradiction_reg, fA)

# make humans
team1.make_human_model(type='logistic',
                       acceptThreshold=accept_criteria,
                       numExamplesToUse=500,
                       numColsToUse=0,
                       biasFactor=0,
                       alterations={'goodRange': [0, 0.5],
                                    'goodProb': 0.5,
                                    'badProb': 0.5,
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
                                    'goodProb': 0.5,
                                    'badProb': 0.5,
                                    'badRange': [0.5, 1],
                                    'Rational': True,
                                    'adder': 0},
                       drop=['humangood'])


# make human good at their region and bad at other region
team1.makeHumanGood()
team3.makeHumanGood()


team2 = deepcopy(team1)

# adjust human confidences
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

team_info.loc[1, 'human true train accepts'] = (
    team1.data_model_dict['train_accept']).sum()
team_info.loc[1, 'human true train rejects'] = (
    ~team1.data_model_dict['train_accept']).sum()
team_info.loc[2, 'human true train accepts'] = (
    team2.data_model_dict['train_accept']).sum()
team_info.loc[2, 'human true train rejects'] = (
    ~team2.data_model_dict['train_accept']).sum()

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

team_info.loc[3, 'human true train accepts'] = (
    team3.data_model_dict['train_accept']).sum()
team_info.loc[3, 'human true train rejects'] = (
    ~team3.data_model_dict['train_accept']).sum()

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

folder = 'gaussian_discretion_results0.3cost'
team_info.to_pickle('{}/start_info.pkl'.format(folder))

team1.data_model_dict['Xtrain'].to_pickle('{}/startDataSet.pkl'.format(folder))

team1_rule_lists = pd.DataFrame(index=range(0, 20), columns=[
                                'TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])
team2_rule_lists = pd.DataFrame(index=range(0, 20), columns=[
                                'TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])
team3_rule_lists = pd.DataFrame(index=range(0, 20), columns=[
                                'TR_prules', 'TR_nrules', 'HyRS_prules', 'HyRS_nrules'])

print('Starting Experiments....... \n')

disc_errors = [0.01, 0.05, 0.25, 0.5, 0.8, 1]
# Repeat Experiments
for disc_error in disc_errors:

    for run in range(0, 10):

        team_info = pd.DataFrame(index=[1, 2, 3])

        coverage_reg = 0
        contradiction_reg = 0.3
        fA = 0.5
        # split training and test randomly
        team1.makeAdditionalTestSplit(testPercent=0.2, replaceExisting=True, random_state=run,
                                      others=[team2, team3])

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

        startData.to_pickle('{}/discError_{}_dataset_run{}.pkl'.format(folder, disc_error, run))

        for i in range(1, 4):
            team_info.loc[i, 'Human Test Acc'] = metrics.accuracy_score(teams[i - 1].data_model_dict['Ytest'],
                                                                        teams[i - 1].data_model_dict['Ybtest'])

        # train aversion and error boundary models
        team1.train_mental_aversion_model('xgboost', probWrong=0, noise=0, data_to_use=disc_error)
        team1.train_mental_error_boundary_model()
        team_info.loc[1, 'human true accepts'] = (team1.data_model_dict['test_conf'] < team1_2_start_threshold).sum()
        team_info.loc[1, 'human true rejects'] = (team1.data_model_dict['test_conf'] >= team1_2_start_threshold).sum()
        team_info.loc[1, 'human accept region test acc'] = metrics.accuracy_score(
            team1.data_model_dict['Ybtest'][team1.data_model_dict['test_accept']],
            team1.data_model_dict['Ytest'][team1.data_model_dict['test_accept']])
        team_info.loc[1, 'human reject region test acc'] = metrics.accuracy_score(
            team1.data_model_dict['Ybtest'][~team1.data_model_dict['test_accept']],
            team1.data_model_dict['Ytest'][~team1.data_model_dict['test_accept']])
        team_info.loc[1, 'aversion model test acc'] = metrics.accuracy_score(
            team1.data_model_dict['paccept_test'] > 0.5,
            team1.data_model_dict['test_accept'])

        team2.train_mental_aversion_model('logistic', probWrong=0, noise=0, data_to_use=disc_error)
        team2.train_mental_error_boundary_model()
        team_info.loc[2, 'human true accepts'] = (team2.data_model_dict['test_conf'] < team1_2_start_threshold).sum()
        team_info.loc[2, 'human true rejects'] = (team2.data_model_dict['test_conf'] >= team1_2_start_threshold).sum()
        team_info.loc[2, 'human accept region test acc'] = metrics.accuracy_score(
            team2.data_model_dict['Ybtest'][team2.data_model_dict['test_accept']],
            team2.data_model_dict['Ytest'][team2.data_model_dict['test_accept']])
        team_info.loc[2, 'human reject region test acc'] = metrics.accuracy_score(
            team2.data_model_dict['Ybtest'][~team2.data_model_dict['test_accept']],
            team2.data_model_dict['Ytest'][~team2.data_model_dict['test_accept']])
        team_info.loc[2, 'aversion model test acc'] = metrics.accuracy_score(
            team2.data_model_dict['paccept_test'] > 0.5,
            team2.data_model_dict['test_accept'])

        team3.train_mental_aversion_model('xgboost', probWrong=0, noise=0, data_to_use=disc_error)
        team3.train_mental_error_boundary_model()
        team_info.loc[3, 'human true accepts'] = (team3.data_model_dict['test_conf'] < team3_4_start_threshold).sum()
        team_info.loc[3, 'human true rejects'] = (team3.data_model_dict['test_conf'] >= team3_4_start_threshold).sum()
        team_info.loc[3, 'human accept region test acc'] = metrics.accuracy_score(
            team3.data_model_dict['Ybtest'][team3.data_model_dict['test_accept']],
            team3.data_model_dict['Ytest'][team3.data_model_dict['test_accept']])
        team_info.loc[3, 'human reject region test acc'] = metrics.accuracy_score(
            team3.data_model_dict['Ybtest'][~team3.data_model_dict['test_accept']],
            team3.data_model_dict['Ytest'][~team3.data_model_dict['test_accept']])
        team_info.loc[3, 'aversion model test acc'] = metrics.accuracy_score(
            team3.data_model_dict['paccept_test'] > 0.5,
            team3.data_model_dict['test_accept'])


        team_info.to_pickle('{}/team_info_dataused{}_run{}.pkl'.format(folder, disc_error, run))



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

        
        if contradiction_reg == 0:
            print('training team1 brs model...')
            team1.setup_brs()
            team1.train_brs()
            
        
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
        
        if contradiction_reg == 0:
            print('training team2 brs model...')
            team2.setup_brs()
            team2.train_brs()
            # print(team2.brs_results['test_error_modelonly'])
        

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
        
        if contradiction_reg == 0:
            print('training team3 brs model...')
            team3.setup_brs()
            team3.train_brs()
            #print(team3.brs_results['test_error_modelonly'])

        

        # write results
        print('writing results...')

        # team1


        team1.full_hyrs_results.to_pickle(
            '{}/discError_{}_team1_hyrs_filtered_run{}.pkl'.format(folder, disc_error, run))
        team1.full_tr_results.to_pickle(
            '{}/discError_{}_team1_tr_filtered_run{}.pkl'.format(folder, disc_error, run))

        with open('{}/discError_{}_team1_hyrs_results_run{}.pkl'.format(folder, disc_error, run), 'wb') as outp:
            pickle.dump(team1.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()

        with open('{}/discError_{}_team1_tr_results_run{}.pkl'.format(folder, disc_error, run), 'wb') as outp:
            pickle.dump(team1.tr_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()



        # team2

        team2.full_hyrs_results.to_pickle(
            '{}/discError_{}_team2_hyrs_filtered_run{}.pkl'.format(folder, disc_error, run))
        team2.full_tr_results.to_pickle(
            '{}/discError_{}_team2_tr_filtered_run{}.pkl'.format(folder, disc_error, run))

        with open('{}/discError_{}_team2_hyrs_results_run{}.pkl'.format(folder, disc_error, run), 'wb') as outp:
            pickle.dump(team2.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()

        with open('{}/discError_{}_team2_tr_results_run{}.pkl'.format(folder, disc_error, run), 'wb') as outp:
            pickle.dump(team2.tr_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()

        # team3

        team3.full_hyrs_results.to_pickle(
            '{}/discError_{}_team3_hyrs_filtered_run{}.pkl'.format(folder, disc_error, run))
        team3.full_tr_results.to_pickle(
            '{}/discError_{}_team3_tr_filtered_run{}.pkl'.format(folder, disc_error, run))

        with open('{}/discError_{}_team3_hyrs_results_run{}.pkl'.format(folder, disc_error, run), 'wb') as outp:
            pickle.dump(team3.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()

        with open('{}/discError_{}_team3_tr_results_run{}.pkl'.format(folder, disc_error, run), 'wb') as outp:
            pickle.dump(team3.tr_results, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()


print('finally done.')