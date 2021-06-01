# -*- coding: utf-8 -*-
"""
Major class: mulParPerAnalysisSim

This class provides the functionality of performance anslysis for
autoencoder-based residual deep network based on the KERAS environment.
The users can set different network configure and training parameters
this class supports parallel computing.

These parameters are included in a dict and include the following items:
    'name': name for a duty;
    'nfea': number of features
    'layernNode': list of integer, Numbers of nodes for each decoding layer
                     (just requirement for the encoding layer)
    'acts':list of strings, activation functions for the hidden layers.
    'extranOutput': extranOutput: integer, output type
            0: target output (1 output for regression or multiple outputs for classification);
            1: target output plus input variables (similar to autoencoder's output) and
                the input variables as partial outputs just like regularizers for the target
                variable.
    'batchnorm': bool, flag for batch normalization (default: True).
    'reg': string, traditional regularizer, like L1 or L2 .
    'inresidual': bool, flag for residual connections (default: False)
    'defact': string, activation function for the output layer (default: None, meaning 'linear')
    'outputtype':integer, 0 or 1, output type. 0 indicates just the target variable
               as the output, 1 indicates the target variable and covariates together as the output
               variables.
    'batch_size': mini batch size.

A typical example:
    tpath='/tmp'
    msim=mulParPerAnalysisSim(tpath)
    msim.getInputSample()

    atask={'name':'TaskR1','nfea':8,'layernNodes':[32,16,8,4], 'acts':'elu', 'extranOutput':1, 'batchnorm':True, 'reg': None,
           'inresidual':True, 'defact':'tanh', 'outputtype':0,'batch_size':128}
    msim.addaDuty(atask)
    for i in range(4000,500,-100):
           atask['name']='bz_'+str(i)
           atask['batch_size']=i
           msim.addaDuty(atask.copy())
    msim.startMProcess(5)

Author: Lianfa Li
Date: 2018-10-01

"""
import pandas as pd
import math
import multiprocessing
from multiprocessing import Process, Manager
import numpy as np
from sklearn import preprocessing

from resautonet.model.resAutoencoder import  resAutoencoder
from resautonet.model import r2K,r2KAuto
from resautonet.model import  rmse


class mulParPerAnalysis:
    def __init__(self,islog=False,outpath='/tmp'):
        """
        Initialization function for Class mulParPerAnalysis
        :param islog: bool, flag for use of log function used for the output variable (default: False)
        :param outpath: string, output path to save the results (default: '/tmp')

        class attribute, tasks: list of string to store the duties
        """
        print('initializing ... ')
        self.tasks=[]
        self.outpath=outpath if outpath is not None else '/tmp'

    def addaDuty(self,aduty):
        """
        Funtion to add a duty of computation corresponding to a config of network
        :param aduty: dict, key-value to indicating the config parameters of the network:
               'name': name for a duty;
                'nfea': number of features
                'layernNode': list of integer, Numbers of nodes for each decoding layer
                                 (just requirement for the encoding layer)
                'acts':list of strings, activation functions for the hidden layers.
                'extranOutput': extranOutput: integer, output type
                        0: target output (1 output for regression or multiple outputs for classification);
                        1: target output plus input variables (similar to autoencoder's output) and
                            the input variables as partial outputs just like regularizers for the target
                            variable.
                'batchnorm': bool, flag for batch normalization (default: True).
                'reg': string, traditional regularizer, like L1 or L2 .
                'inresidual': bool, flag for residual connections (default: False)
                'defact': string, activation function for the output layer (default: None, meaning 'linear')
                'outputtype':integer, 0 or 1, output type. 0 indicates just the target variable
                           as the output, 1 indicates the target variable and covariates together as the output
                           variables.
                'batch_size': mini batch size.
        """
        self.tasks.append(aduty)

    def getInputSample(self,rPath,datafl,spSampleFl):
        """
        Function to get the input data samples
        :param rPath: string, root path for the data files
        :param datafl: string, Original CSV format (pandas DataFrame) data file
        :param spSampleFl:string, the file for the numpy array compressed arrays,
               including x_train, y_train,x_valid, y_valid, x_test,y_test
               This is to make sure the data split for training
        """
        loaded = np.load(rPath+"/"+datafl)
        y = loaded['y']
        self.scy = preprocessing.StandardScaler().fit(y)
        loaded=np.load(rPath+"/"+spSampleFl)
        self.x_train = loaded['x_train']
        self.y_train = loaded['y_train']
        self.x_valid = loaded['x_valid']
        self.y_valid = loaded['y_valid']
        self.x_test = loaded['x_test']
        self.y_test = loaded['y_test']


    def subTrain(self, perm,istart, iend):
        """
        Function for a subtrain process
        :param perm: list to save the result of this training function
        :param istart: start index for the attribute, tasks.
        :param iend: end index for the attribute, tasks.
        """
        p = multiprocessing.current_process()
        # print("Starting process "+p.name+", pid="+str(p.pid)+" ... ...")
        nduty = iend - istart
        for i in range(istart, iend):
            atask=self.tasks[i]
            modelCls = resAutoencoder(**atask)
            model = modelCls.resAutoNet()
            model.summary()
            model.compile(optimizer="adam", loss='mean_squared_error',metrics=['mean_squared_error', r2KAuto, r2K])
            fhist = model.fit(self.x_train, self.y_train, batch_size=atask['batch_size'], epochs=100, verbose=0, shuffle=True,
                              validation_data=(self.x_valid, self.y_valid))
            y_test_pred = model.predict(self.x_test)

            obs = self.scy.inverse_transform(self.y_test[:, -1])
            pre = self.scy.inverse_transform(y_test_pred[:, -1])
            r2 = rsquared(obs, pre)
            rmse = rmse(obs, pre)
            print("indepdendent test:r2-", r2, "rmse:", rmse)
            #datafl = self.outpath + '/res_'+atask['name']+'.npz'
            ares=pd.DataFrame({'name':atask['name'],'max_reg_r2':max(fhist.history['r2KAuto']),'min_reg_rmse':min(fhist.history['loss']),
                    'max_val_r2':max(fhist.history['val_r2KAuto']),'min_val_rmse':min(fhist.history['val_loss']),
                    'testr2':r2,'testrmse':rmse},index=[0])
            perm.append(ares)
            tPath = self.outpath+'/s_' + atask['name'] + '.csv'
            pd.DataFrame(fhist.history).to_csv(tPath)

        print("Done with " + p.name + ", pid=" + str(p.pid) + "!")

    def startMProcess(self, ncore):
        """
        Function to initiate the parallel computing for the set of duties for performance analysis
        :param ncore: number of CPU processes to be used in training
        :return: the result to be saved in the file
        """
        n = len(self.tasks)
        nTime = int(math.ceil(n / ncore))
        print(str(ncore) + " cores for " + str(n) + " duties; each core has about " + str(nTime) + " duties")
        manager = Manager()
        perm= manager.list()
        for t in range(0, nTime):
            istart = t * ncore
            iend = (t + 1) * ncore
            if t == (nTime - 1):
                iend = n
            processes = []
            for k in range(istart, iend):
                p = Process(name=str(t), target=self.subTrain, args=(perm,k, k + 1,))
                p.daemon = True
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        datafl = self.outpath + '/al1res.csv'
        allres=pd.concat(perm, axis=0)
        allres.to_csv(datafl,index_label='index')
