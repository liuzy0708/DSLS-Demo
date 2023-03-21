# Imports
from Get_JiaoLong_Data import get_Jiaolong_Data_noise_30, get_Jiaolong_Data_noise_90, get_Jiaolong_Data_noise_150
import skmultiflow as sk
import numpy as np
import random
from skmultiflow.data import SEAGenerator, SineGenerator, AGRAWALGenerator
from skmultiflow.data import HyperplaneGenerator, WaveformGenerator
import warnings
from Get_para_init import DSAAI_Config, Rand_Config, CogDQS_Config, DMGT_Config, ROALE_DI_Config, All_Exp_Config
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from Get_model_init import model_init
from AVI_tools import set_log_file
from Def_DSA_AI import DSA_AI_strategy
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from DES import DES_ICD
import csv
import time

t1 = time.time()
logpath = "run.log"
set_log_file(logpath)
warnings.filterwarnings("ignore")

n_class = 3
n_round = 3

DSLS_DSA_AI_str = DSA_AI_strategy(memory_collection=[[i for i in range(0, n_inital_30)]] + [[i for i in range(n_inital_30, 2*n_inital_30)]] + [[i for i in range(2*n_inital_30, 3*n_inital_30)]])

max_samples = 10000  # The range of tested stream

All_Exp_para = All_Exp_Config()

for main_loop in range(n_round):

    DSA_AI_para = DSAAI_Config()

    # Setup a data stream
    # stream = SEAGenerator(random_state=1)
    stream = WaveformGenerator(random_state=1)

    # Setup Classifier

    X_pretrain, y_pretrain = stream.next_sample(30)


    DSLS, \
    oselm_CogDQS, ob_CogDQS, ht_CogDQS, arf_CogDQS, efdt_CogDQS, srp_CogDQS, des_CogDQS, \
    oselm_ROALE_DI, ob_ROALE_DI, ht_ROALE_DI, arf_ROALE_DI, efdt_ROALE_DI, srp_ROALE_DI, des_ROALE_DI = model_init()

    'DSLS'
    DSLS.fit(X_pretrain, y_pretrain)

    count = 0
    result = []

    # Setup variables to control loop and track performance

    # X, y = stream.next_sample(20000)

    # Train the classifier with the samples provided by the data stream

    while count < max_samples and stream.has_more_samples():

        X, y = stream.next_sample()

        DSA_AI_para.y_pred_DSLS = DSLS.predict(X)

        result = result + [y[0]]

        DSA_AI_para.result_DSLS = DSA_AI_para.result_DSLS + [DSA_AI_para.y_pred_DSLS[0]]

        'DSA-AI Strategy (clf更新 + 标注统计)'

        DSLS, _ = DSLS_DSA_AI_str.DSA_AI_evaluation(X, y, DSLS, count, data, label, n_class=n_class)
        count += 1


        if count % (max_samples * 0.10) == 0:
            print('\nHave processed {:.0f}%'.format(count / max_samples * 100), 'samples')


    print('------------------------------------n_annotation-----------------------------------------------------------')
    print('annotation_DSLS = {}'.format(All_Exp_para.n_annotation_DSLS_list))


    with open(r'predict_result_3_14_90_DSLS{}.csv'.format(main_loop), 'w', encoding='utf-8', newline='') as fp:
        print('begin')
        writer = csv.writer(fp)
        writer.writerow(result)
        writer.writerow(DSA_AI_para.result_DSLS)
        print('done')

    # All_Exp_para.n_annotation_DSLS = All_Exp_para.n_annotation_DSLS + DSLS_DSA_AI_str.n_annotation

'Display results'

All_Exp_para.DSLS_acc = np.array(All_Exp_para.DSLS_acc).reshape(n_round, Interval_size)
All_Exp_para.DSLS_f1 = np.array(All_Exp_para.DSLS_f1).reshape(n_round, Interval_size)
All_Exp_para.n_annotation_DSLS_list = np.array(All_Exp_para.n_annotation_DSLS_list).reshape(n_round, Interval_size)
print(All_Exp_para.DSLS_acc, All_Exp_para.DSLS_f1)


