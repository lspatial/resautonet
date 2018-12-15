# -*- coding: utf-8 -*-
"""
Major class: resAutoencoder

This class provides the functionality of autoencoder-based residual
deep network based on the KERAS environment. The users just inputs
their arguments including number of nodes for each decoding layer,
output type, initializer, dropout rate, flag for batch normalization,
regularizer, flag for residual connection, activation functions and
output type to obtain a residual network. We provide two versions for
recursion and iteration.

Author: Lianfa Li
Date: 2018-10-01

"""

from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout,Dense, Activation,add

class resAutoencoder:

    def __init__(self,nfea,layernNodes,acts=None, extranOutput=0, k_initializer='he_normal',dropout=0.5,
                 batchnorm=True,reg=None,inresidual=False,outnres=None,defact=None,outputtype=0,*args, **kwargs):
        """
        Initialization function to set up the autoencoder-based residual deep network
        :param nfea: integer, number of features
        :param layernNodes: list of integer, Numbers of nodes for each decoding layer
                 (just requirement for the encoding layer)
        :param acts: list of strings, activation functions for the hidden layers (defaults: relu).
        :param extranOutput: integer, the number of the target variables (default: 0)

        :param k_initializer: string, initialization functions, see Keras documents for details
               (default:'he_normal').
        :param dropout: double,0-1, dropout rate, just applied at the middle coding layer (default: 0.5).
        :param batchnorm: bool, flag for batch normalization (default: True).
        :param reg: string, traditional regularizer, like L1 or L2 (default: None).
        :param inresidual: bool, flag for residual connections (default: False)
        :param outnres: integer, number of output nodes (1 for regression; 2 or more for classification)
                (default: None, indicating 1 for regression).
        :param defact: string, activation function for the output layer (default: None, meaning 'linear').
        :param outputtype: integer, 0 or 1, output type (default: 0).
                0: target output (1 output for regression or multiple outputs for classification);
                1: target output plus input variables (similar to autoencoder's output) and
                the input variables as partial outputs just like regularizers for the target variable.
        :param args: other parameters
        :param kwargs: other key parameters
        """
        self.nfea=nfea
        self.extranOutput=extranOutput
        self.layernNodes=layernNodes
        self.defact=defact
        if acts is None:
            self.acts=['relu' for i in range(len(layernNodes))]
        elif isinstance(acts,str):
            self.acts = [acts for i in range(len(layernNodes))]
        elif isinstance(acts, list)  and len(acts)==len(layernNodes):
            self.acts=acts
        else:
            self.acts =['linear' for i in range(len(layernNodes))]
        self.dropout=dropout
        self.batchnorm=batchnorm
        self.nmaxpool=-1
        self.k_initializer=k_initializer
        self.inresidual=inresidual
        self.outnres=outnres
        self.reg=reg if reg is not None else None # keras.regularizers.l1_l2(0) #keras.regularizers.l1_l2(0.)
        self.outputtype=outputtype

    def getActivation(self,actstr,inlayer):
        """
        Function to obtain the activation layer
        :param actstr: string, action function. default: None, indicating 'linear'
        :param inlayer: Input layer
        :return: the activation layer's output
        """
        prereact=['relu','elu','softmax','selu','softplus','tanh','sigmoid','linear','hard_sigmoid']
        #preadvact={'LeakyReLU':LeakyReLU()(inlayer)}
        if actstr is None:
            return inlayer
        if actstr in prereact:
            return Activation(actstr)(inlayer)
        else:
            return eval(actstr+'()(inlayer)')
        return inlayer

    def hiddenBlock(self, inlayer, idepth, orginlayer=None,dropout=0):
        """
        Function for resursive block of the hidden layer
        :param inlayer: input layer
        :param idepth: integer, index for the depth (of the decoding layer).
        :param orginlayer: the original encoding layer to be used as
               residual connection (default: None, indicating decoding
               layer or no residual connection)
        :param dropout: double, dropout rate for the middle coding layer
                (default 0).
        :return: output layer
        """
        layer=Dense(self.layernNodes[idepth],kernel_initializer='glorot_uniform', kernel_regularizer=self.reg)(inlayer)
#        if orginlayer is not None:
#            layer = add([orginlayer, layer])
        layer = BatchNormalization()(layer) if self.batchnorm else layer
        layer=self.getActivation(self.acts[idepth],layer)
        layer = BatchNormalization()(layer) if self.batchnorm else layer
        layer = Dropout(dropout)(layer) if dropout else layer
        if orginlayer is not None :
             layer = add([orginlayer, layer])
             layer = BatchNormalization()(layer) if self.batchnorm else layer
             layer = self.getActivation(self.acts[idepth], layer)
        return layer

    def levelsBlock(self, inlayer, idepth, dropout):
        """
        Function for the level block in the nested way
        :param inlayer:  input layer
        :param idepth: idepth: integer, index for the depth (of the decoding layer).
        :param dropout: dropout: double, dropout rate for the middle coding layer.
        :return: output layer
        """
        if idepth < (len(self.layernNodes)-1):
            layer = self.hiddenBlock(inlayer, idepth)
            mlayer = self.levelsBlock(layer, idepth + 1, dropout)
            mlayer = self.hiddenBlock(mlayer,idepth,layer if self.inresidual else None)
        else:
            mlayer = self.hiddenBlock(inlayer,idepth,None,dropout)
        return mlayer

    def basicResUnit(self,inlayer,iresunit):
        """
        Function to construct one basic residual network to construct
            multiple levels of residual deep networks
        :param inlayer: input layer, input layer or the output layer of
                 last residual network
        :param iresunit: integer, 0: residual network, 1: regular network
        :return: output layer for the basic residual network
        """
        if iresunit>1:
            unitoutput=self.levelsBlock(inlayer,0,self.dropout)
            unitoutput=Dense(self.nfea, kernel_initializer='he_normal',
                  kernel_regularizer=self.reg)(unitoutput)
            unitoutput=add([inlayer, unitoutput])
            # unitoutput=concatenate([inlayer, unitoutput], -1)
            unitoutput=self.basicResUnit(unitoutput,iresunit-1)
        else:
            unitoutput = self.levelsBlock(inlayer, 0, self.dropout)
        return unitoutput

    def resAutoNet(self):
        """
        Function to construct autoencoder-based (residual) deep network
          depending on the class argument, inresidual
               (True: residual network, False: regular network)
        implemented using the recursive functions.
        :return: the network model constructed.
        """
        inputlayer = Input(shape=(self.nfea,),name='feat')
        if self.outnres is None or self.outnres==0:
            layer = self.levelsBlock(inputlayer,0, self.dropout)
        else:
            layer = self.basicResUnit(inputlayer, self.outnres)
        #outlayer = Dense(self.nfea + self.extranOutput, kernel_initializer='he_normal',
        if self.outputtype==2:
            outlayer = Dense(self.nfea + self.extranOutput, kernel_initializer='he_normal',
                             kernel_regularizer=self.reg)(layer)
            return Model(inputs=inputlayer, outputs=outlayer)
        layer = Dense(self.nfea, kernel_initializer='he_normal', kernel_regularizer=self.reg)(layer)
        layer = self.getActivation(self.acts[len(self.acts) - 1], layer)
        layer = BatchNormalization()(layer) if self.batchnorm else layer
        if self.inresidual:
            layer = add([inputlayer, layer])
            layer = self.getActivation(self.defact, layer)
            layer = BatchNormalization()(layer) if self.batchnorm else layer
        if self.outputtype == 0:
            outlayer = Dense(self.extranOutput,activation=self.defact, kernel_initializer='he_normal',
                             kernel_regularizer=self.reg)(layer)
        else:
            outlayer = Dense(self.nfea+self.extranOutput,activation=self.defact, kernel_initializer='he_normal',
                             kernel_regularizer=self.reg)(layer)
        return Model(inputs=inputlayer, outputs=outlayer)


    def resAutoNetIt(self):
        """
        Function to construct autoencoder-based (residual) deep network
          depending on the class argument, inresidual
               (True: residual network, False: regular network)
          implemented the same functionality as resAutoNet but using the
          iteration way.
        :return:the network model constructed.
        """
        inputlayer = Input(shape=(self.nfea,),name='feat')
        stck=[]
        layer=inputlayer
        for i in range(len(self.layernNodes)):
            nnode=self.layernNodes[i]
            layer = Dense(nnode, kernel_initializer='he_normal', kernel_regularizer=self.reg)(layer)
            layer = self.getActivation(self.acts[i], layer)
            layer = BatchNormalization()(layer) if self.batchnorm else layer
            if i<(len(self.layernNodes)-1):
                stck.append(layer)
            if i==(len(self.layernNodes)-1):
                layer = Dropout(self.dropout)(layer) if self.dropout else layer
        for i in range(len(self.layernNodes)-2,-1,-1):
            orlayer=stck.pop()
            layer = self.hiddenBlock(layer, i,orlayer)
        layer = Dense(self.nfea, kernel_initializer='he_normal', kernel_regularizer=self.reg)(layer)
        layer = self.getActivation(self.acts[len(self.acts)-1], layer)
        layer = BatchNormalization()(layer) if self.batchnorm else layer
        if self.inresidual :
            layer = add([inputlayer, layer])
            layer = BatchNormalization()(layer) if self.batchnorm else layer
            layer = self.getActivation(self.defact, layer)
        outlayer = Dense(self.extranOutput, kernel_initializer='he_normal',
                         kernel_regularizer=self.reg)(layer)
        return Model(inputs=inputlayer, outputs=outlayer)

    def getSimpleNet(self):
        """
        Function to construct simple network without hidden layers for
          the purpose of comparison with residual networks.
        :return: the network model constructed.
        """
        inputlayer = Input(shape=(self.nfea,), name='feat')
        layer = BatchNormalization()(inputlayer) if self.batchnorm else inputlayer
        outlayer = Dense(self.extranOutput, kernel_initializer='he_normal',
                         kernel_regularizer=self.reg)(layer)
        return Model(inputs=inputlayer, outputs=outlayer)