#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:30:41 2019

@author: tyleryoshihara 
"""

import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib, scipy, sklearn, parfit, scikitplot, pandas, nilearn
from sklearn import preprocessing, model_selection, linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nipype.interfaces import matlab
import matplotlib.pyplot as plt
import pandas
import numpy as np

HCvMCI = pandas.read_csv('/Users/tyleryoshihara/Desktop/neuro182/finalproject/MCIvsHCFourier.csv')
MCIvAD = pandas.read_csv('/Users/tyleryoshihara/Desktop/neuro182/finalproject/MCIvsADFourier.csv')
ADvHC = pandas.read_csv('/Users/tyleryoshihara/Desktop/neuro182/finalproject/ADvsHCFourier.csv')

HCvMCI = np.asarray(HCvMCI)
MCIvAD = np.asarray(MCIvAD)
ADvHC = np.asarray(ADvHC)

HCvMCI = HCvMCI[:,1:]
MCIvAD = MCIvAD[:,1:]
ADvHC = ADvHC[:,1:]

YHvM = HCvMCI[:,304]
YMvA = MCIvAD[:,304]
YAvH = ADvHC[:,304]

HCvMCI = HCvMCI[:,0:304]
MCIvAD = MCIvAD[:,0:304]
ADvHC = ADvHC[:,0:304]

HCvMCI = preprocessing.scale(HCvMCI, axis=0)
MCIvAD = preprocessing.scale(MCIvAD, axis=0)
ADvHC = preprocessing.scale(ADvHC, axis=0)

freq_bands = scipy.signal.welch(ADvHC, fs=256)

#All models for HCvMCI
# Set up Train, Val, Test sets for all datasets
[train_inds, test_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.2,random_state=100).split(
                        HCvMCI,y=YHvM
                        )
        )

XTempTrain = HCvMCI[train_inds,]

# Split Train Set into Train and Validation Sets
[train2_inds, val_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.4,random_state=100).split(
                        XTempTrain,y=YHvM[train_inds]
                        )
        )

# Form the indices to select for each set
TrainInds = train_inds[train2_inds]
ValInds = train_inds[val_inds]
TestInds = test_inds


# Create sets of X and Y data using indices  for HCvMCI
    
XTrainHvM = HCvMCI[TrainInds,]
YTrainHvM = YHvM[TrainInds]
XValHvM = HCvMCI[ValInds,]
YValHvM = YHvM[ValInds]
XTestHvM = HCvMCI[TestInds,]
YTestHvM = YHvM[TestInds]

#Running RVC - HCvMCI

RXTrainHvM = HCvMCI[train_inds,]
RYTrainHvM = YHvM[train_inds]
RXTestHvM = HCvMCI[test_inds,]
RYTestHvM = YHvM[test_inds]

from skrvm import RVC
RVCMod = RVC(kernel = 'linear',
             verbose = True)
RVCMod.fit(RXTrainHvM,RYTrainHvM)

def RVMFeatImp(RVs):
    NumRVs = RVs.shape[0]
    SumD = 0
    for RVNum in range(1,NumRVs):
        d1 = RVs[RVNum-1,]
        d2 = sum(np.ndarray.flatten(
                RVs[np.int8(
                        np.setdiff1d(np.linspace(0,NumRVs-1,NumRVs),RVNum))]))
        SumD = SumD + (d1/d2)
    SumD = SumD/NumRVs
    return SumD


RVs = RVCMod.relevance_
DVals = RVMFeatImp(RVs)

RVCPred1 = RVCMod.predict_proba(RXTestHvM)
RVCPred2 = RVCMod.predict(RXTestHvM)
# Evaluate Performance (DON'T RELY ON ACCURACY!!!)
# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(RYTestHvM,RVCPred1, title = 'HCvMCI: RVC')
# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(RYTestHvM,RVCPred2)

#%%
# Running RLR - HCvMCI

#Testing for multicollinearity 

coef1 = np.corrcoef(HCvMCI, rowvar = False)
plt.hist(coef1)

coef2 = np.corrcoef(MCIvAD, rowvar = False)
plt.hist(coef2)

coef3 = np.corrcoef(ADvHC, rowvar = False)
plt.hist(coef3)

ncores = 2
grid = {
    'C': np.linspace(1e-10,1e5,num = 100), #Inverse lambda
    'penalty': ['l1']
}
paramGrid = sklearn.model_selection.ParameterGrid(grid)
RLRMod = sklearn.linear_model.LogisticRegression(tol = 1e-10,
                                                random_state = 100,
                                                n_jobs = ncores,
                                                verbose = 1)
[bestModel,bestScore,allModels,allScores] = parfit.bestFit(RLRMod,
paramGrid = paramGrid,               
X_train = XTrainHvM,
y_train = YTrainHvM,
X_val = XValHvM,
y_val = YValHvM,
metric = sklearn.metrics.roc_auc_score,
n_jobs = ncores,
scoreLabel = 'AUC')

# Test on Test Set
RLRTestPred = bestModel.predict_proba(XTestHvM)
RLRTestPred2 = bestModel.predict(XTestHvM)

# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestHvM,RLRTestPred,title = 'LR with LASSO')

# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestHvM,RLRTestPred2)


# %%
# RF - HCvMCI

# Define the Hyperparameter grid values
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Code for the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

# Declare the Classifier
RFC = RandomForestClassifier(criterion = 'gini') #can use 'entropy' instead

# Raw Classifier

from sklearn.model_selection import RandomizedSearchCV
# n_iter is the number of randomized parameter combinations tried
RFC_RandomSearch = RandomizedSearchCV(estimator = RFC,
                                      param_distributions = random_grid,
                                      n_iter = 100, cv = 3, verbose=2,
                                      random_state=10, n_jobs = 2)
RFC_RandomSearch.fit(XTrainHvM,YTrainHvM)

# Look at the Tuned "Best" Parameters
RFC_RandomSearch.best_params_
RFC_RandomSearch.best_score_
RFC_RandomSearch.best_estimator_.feature_importances_

# Fit using the best parameters
# Look at the Feature Importance
FeatImp = RFC_RandomSearch.best_estimator_.feature_importances_
NZInds = np.nonzero(FeatImp)
num_NZInds = len(NZInds[0])
Keep_NZVals = [x for x in FeatImp[NZInds[0]] if 
               (abs(x) >= np.mean(FeatImp[NZInds[0]]) 
               + 4*np.std(FeatImp[NZInds[0]]))]
ThreshVal = np.mean(FeatImp[NZInds[0]]) + 2*np.std(FeatImp[NZInds[0]])
Keep_NZInds = np.nonzero(abs(FeatImp[NZInds[0]]) >= ThreshVal)
Final_NZInds = NZInds[0][Keep_NZInds]



Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestHvM)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestHvM)

scikitplot.metrics.plot_roc(YTestHvM,Pred2_S2, title = 'HCvMC RF')
scikitplot.metrics.plot_confusion_matrix(YTestHvM,Pred1_S2)

#FD for HvM
[XTrainFDHvM,YTrainFDHvM] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainHvM,YTrainHvM)
[XTestFDHvM,YTestFDHvM] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestHvM,YTestHvM)

FDMod = LinearDiscriminantAnalysis(tol = 1e-4,solver = 'svd')
FDFit = FDMod.fit(XTrainFDHvM,YTrainFDHvM)

FIvec_FD_HvM = FDMod.coef_

FDTestPredHvM = FDFit.predict_proba(XTestFDHvM)
FDTestPred2HvM = FDFit.predict(XTestFDHvM)
scikitplot.metrics.plot_roc(YTestFDHvM,FDTestPredHvM ,title = 'FLD')
scikitplot.metrics.plot_confusion_matrix(YTestFDHvM,FDTestPred2HvM)

accuracy = (FDTestPred2MvA == YTestFDMvA).mean() * 100.
print("FDA classification accuracy : %g%%" % accuracy) 




#All models for MCI v AD
# Set up Train, Val, Test sets for MCI vs AD
[train_inds, test_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.2,random_state=100).split(
                        MCIvAD,y=YMvA
                        )
        )

XTempTrain = MCIvAD[train_inds,]

# Split Train Set into Train and Validation Sets
[train2_inds, val_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.4,random_state=100).split(
                        XTempTrain,y=YMvA[train_inds]
                        )
        )

# Form the indices to select for each set
TrainInds = train_inds[train2_inds]
ValInds = train_inds[val_inds]
TestInds = test_inds


# Create sets of X and Y data using indices  for MCIvAD
    
XTrainMvA = MCIvAD[TrainInds,]
YTrainMvA = YMvA[TrainInds]
XValMvA = MCIvAD[ValInds,]
YValMvA = YMvA[ValInds]
XTestMvA = MCIvAD[TestInds,]
YTestMvA = YMvA[TestInds]

#%%
#Running RVC - MCIvAD

RXTrainMvA = MCIvAD[train_inds,]
RYTrainMvA = YMvA[train_inds]
RXTestMvA = MCIvAD[test_inds,]
RYTestMvA = YMvA[test_inds]

from skrvm import RVC
RVCMod = RVC(kernel = 'linear',
             verbose = True)
RVCMod.fit(RXTrainMvA,RYTrainMvA)

def RVMFeatImp(RVs):
    NumRVs = RVs.shape[0]
    SumD = 0
    for RVNum in range(1,NumRVs):
        d1 = RVs[RVNum-1,]
        d2 = sum(np.ndarray.flatten(
                RVs[np.int8(
                        np.setdiff1d(np.linspace(0,NumRVs-1,NumRVs),RVNum))]))
        SumD = SumD + (d1/d2)
    SumD = SumD/NumRVs
    return SumD


RVs = RVCMod.relevance_
DVals = RVMFeatImp(RVs)

RVCPred1 = RVCMod.predict_proba(RXTestMvA)
RVCPred2 = RVCMod.predict(RXTestMvA)
# Evaluate Performance (DON'T RELY ON ACCURACY!!!)
# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(RYTestMvA,RVCPred1, title = 'MCIvAD: RVC')
# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(RYTestMvA,RVCPred2)

#%%
# Running RLR - MCI vAD

#Testing for multicollinearity 

coef2 = np.corrcoef(MCIvAD, rowvar = False)
plt.hist(coef2)

ncores = 2
grid = {
    'C': np.linspace(1e-10,1e5,num = 100), #Inverse lambda
    'penalty': ['l1']
}
paramGrid = sklearn.model_selection.ParameterGrid(grid)
RLRMod = sklearn.linear_model.LogisticRegression(tol = 1e-10,
                                                random_state = 100,
                                                n_jobs = ncores,
                                                verbose = 1)
[bestModel,bestScore,allModels,allScores] = parfit.bestFit(RLRMod,
paramGrid = paramGrid,               
X_train = XTrainMvA,
y_train = YTrainMvA,
X_val = XValMvA,
y_val = YValMvA,
metric = sklearn.metrics.roc_auc_score,
n_jobs = ncores,
scoreLabel = 'AUC')

# Test on Test Set
RLRTestPred = bestModel.predict_proba(XTestMvA)
RLRTestPred2 = bestModel.predict(XTestMvA)

# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestMvA,RLRTestPred,title = 'LR with LASSO')

# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestMvA,RLRTestPred2)


# %%
# RF - MCIvAD

# Define the Hyperparameter grid values
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Code for the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

# Declare the Classifier
RFC = RandomForestClassifier(criterion = 'gini') #can use 'entropy' instead

# Raw Classifier

from sklearn.model_selection import RandomizedSearchCV
# n_iter is the number of randomized parameter combinations tried
RFC_RandomSearch = RandomizedSearchCV(estimator = RFC,
                                      param_distributions = random_grid,
                                      n_iter = 100, cv = 3, verbose=2,
                                      random_state=10, n_jobs = 2)
RFC_RandomSearch.fit(XTrainMvA,YTrainMvA)

# Look at the Tuned "Best" Parameters
RFC_RandomSearch.best_params_
RFC_RandomSearch.best_score_
RFC_RandomSearch.best_estimator_.feature_importances_

# Fit using the best parameters
# Look at the Feature Importance
FeatImp = RFC_RandomSearch.best_estimator_.feature_importances_
NZInds = np.nonzero(FeatImp)
num_NZInds = len(NZInds[0])
Keep_NZVals = [x for x in FeatImp[NZInds[0]] if 
               (abs(x) >= np.mean(FeatImp[NZInds[0]]) 
               + 4*np.std(FeatImp[NZInds[0]]))]
ThreshVal = np.mean(FeatImp[NZInds[0]]) + 2*np.std(FeatImp[NZInds[0]])
Keep_NZInds = np.nonzero(abs(FeatImp[NZInds[0]]) >= ThreshVal)
Final_NZInds = NZInds[0][Keep_NZInds]



Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestMvA)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestMvA)

scikitplot.metrics.plot_roc(YTestMvA,Pred2_S2, title = 'MCIvAD RF')
scikitplot.metrics.plot_confusion_matrix(YTestMvA,Pred1_S2)

from scipy import stats

FeatImp_RF_MvA_reshape = np.reshape(FeatImp,[19,16])
FeatImp_RF_mean = np.mean(FeatImp_RF_MvA_reshape, axis=0)
FeatImp_RF_std = np.std(FeatImp_RF_MvA_reshape, axis=0)
conf_int = stats.norm.interval(0.95, loc=FeatImp_RF_mean, scale=FeatImp_RF_std)

Freq_values = np.linspace(0,30,16)
plt.plot(Freq_values,FeatImp_RF_mean, 'o')
plt.title("RF Feature Importance by Channel")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean Feature Importance")
plt.show()

# FDA
# Recover Classes Using Fisher's Linear Discriminant Analysis with SVD
[XTrainFDMvA,YTrainFDMvA] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainMvA,YTrainMvA)
[XTestFDMvA,YTestFDMvA] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestMvA,YTestMvA)

FDMod = LinearDiscriminantAnalysis(tol = 1e-4,solver = 'svd')
FDFit = FDMod.fit(XTrainFDMvA,YTrainFDMvA)

FIvec_FD_MvA = FDMod.coef_

FDTestPredMvA = FDFit.predict_proba(XTestFDMvA)
FDTestPred2MvA = FDFit.predict(XTestFDMvA)
scikitplot.metrics.plot_roc(YTestFDMvA,FDTestPredMvA ,title = 'FLD')
scikitplot.metrics.plot_confusion_matrix(YTestFDMvA,FDTestPred2MvA)

accuracy = (FDTestPred2MvA == YTestFDMvA).mean() * 100.
print("FDA classification accuracy : %g%%" % accuracy) 

FIvec_FD_MvA = abs(FIvec_FD_MvA)
FIvec_FD_MvA =FIvec_FD_MvA.T
FIvec_FD_MvA = FIvec_FD_MvA[:,0]


FeatMatrix = np.stack([FeatImp, FIvec_FD_MvA], axis = 1)
FeatCorr = np.corrcoef(FeatMatrix.T)
np.triu(FeatCorr)

plt.plot(FeatImp, FIvec_FD_MvA, 'o')


FeatImp_FDA_MvA_reshape = np.reshape(FIvec_FD_MvA,[19,16])
FeatImp_FDA_mean = np.mean(FeatImp_FDA_MvA_reshape, axis=0)

plt.plot(Freq_values,FeatImp_FDA_mean, 'o')
plt.title("FD Feature Importance by Channel")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean Feature Importance")
plt.show()

np.savetxt("FeatImp_FDA_MvA.csv", FIvec_FD_MvA)
np.savetxt("FeatImp_RF_MvA.csv", FeatImp)

#All models for AD vs HC
# Set up Train, Val, Test sets for AD vs HC
[train_inds, test_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.2,random_state=100).split(
                        ADvHC,y=YAvH
                        )
        )

XTempTrain = MCIvAD[train_inds,]

# Split Train Set into Train and Validation Sets
[train2_inds, val_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.4,random_state=100).split(
                        XTempTrain,y=YAvH[train_inds]
                        )
        )

# Form the indices to select for each set
TrainInds = train_inds[train2_inds]
ValInds = train_inds[val_inds]
TestInds = test_inds

# Create sets of X and Y data using indices  for ADvHC
    
XTrainAvH = ADvHC[TrainInds,]
YTrainAvH = YAvH[TrainInds]
XValAvH = ADvHC[ValInds,]
YValAvH = YAvH[ValInds]
XTestAvH = ADvHC[TestInds,]
YTestAvH = YAvH[TestInds]


#%%
#Running RVC - ADvHC

from imblearn.over_sampling import SMOTE, ADASYN


RXTrainAvH = ADvHC[train_inds,]
RYTrainAvH = YAvH[train_inds]
RXTestAvH = ADvHC[test_inds,]
RYTestAvH = YAvH[test_inds]

[XTrainResAvH,YTrainResAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(RXTrainAvH,RYTrainAvH)
[XTestResAvH,YTestResAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(RXTestAvH,RYTestAvH)


from skrvm import RVC
RVCMod = RVC(kernel = 'linear',
             verbose = True)
RVCMod.fit(RXTrainAvH,RYTrainAvH)

def RVMFeatImp(RVs):
    NumRVs = RVs.shape[0]
    SumD = 0
    for RVNum in range(1,NumRVs):
        d1 = RVs[RVNum-1,]
        d2 = sum(np.ndarray.flatten(
                RVs[np.int8(
                        np.setdiff1d(np.linspace(0,NumRVs-1,NumRVs),RVNum))]))
        SumD = SumD + (d1/d2)
    SumD = SumD/NumRVs
    return SumD


RVs = RVCMod.relevance_
DVals = RVMFeatImp(RVs)

RVCPred1 = RVCMod.predict_proba(XTestResAvH)
RVCPred2 = RVCMod.predict(XTestResAvH)
# Evaluate Performance 
# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestResAvH,RVCPred1, title = 'ADvHC: RVC')
# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestResAvH,RVCPred2)

accuracy = (RVCPred2 == YTestResAvH).mean() * 100.
print("RVC classification accuracy : %g%%" % accuracy) 

#%%
# Running RLR - HCvMCI

#Testing for multicollinearity 

coef3 = np.corrcoef(ADvHC, rowvar = False)
plt.hist(coef3)

[XTrainRLRAvH,YTrainRLRAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainAvH,YTrainAvH)
[XValRLRAvH,YValRLRAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XValAvH,YValAvH)
[XTestRLRAvH,YTestRLRAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestAvH,YTestAvH)


ncores = 2
grid = {
    'C': np.linspace(1e-10,1e5,num = 100), #Inverse lambda
    'penalty': ['l2']
}
paramGrid = sklearn.model_selection.ParameterGrid(grid)
RLRMod = sklearn.linear_model.LogisticRegression(tol = 1e-10,
                                                random_state = 100,
                                                n_jobs = ncores,
                                                verbose = 1)
[bestModel,bestScore,allModels,allScores] = parfit.bestFit(RLRMod,
paramGrid = paramGrid,               
X_train = XTrainRLRAvH,
y_train = YTrainRLRAvH,
X_val = XValRLRAvH,
y_val = YValRLRAvH,
metric = sklearn.metrics.roc_auc_score,
n_jobs = ncores,
scoreLabel = 'AUC')

# Test on Test Set
RLRTestPred = bestModel.predict_proba(XTestRLRAvH)
RLRTestPred2 = bestModel.predict(XTestRLRAvH)

feat_imp_RLR = bestModel.coef_


# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestRLRAvH,RLRTestPred,title = 'AD vs HC with Ridge')

# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestRLRAvH,RLRTestPred2)

accuracy = (RLRTestPred2 == YTestRLRAvH).mean() * 100.
print("FDA classification accuracy : %g%%" % accuracy) #57.1429%



# %%
# RF - AD vs HC

# Define the Hyperparameter grid values
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Code for the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

# Declare the Classifier
RFC = RandomForestClassifier(criterion = 'gini') #can use 'entropy' instead

[XTrainRFAvH,YTrainRFAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainAvH,YTrainAvH)
[XValRFAvH,YValRFAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XValAvH,YValAvH)
[XTestRFAvH,YTestRFAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestAvH,YTestAvH)

# Raw Classifier

from sklearn.model_selection import RandomizedSearchCV
# n_iter is the number of randomized parameter combinations tried
RFC_RandomSearch = RandomizedSearchCV(estimator = RFC,
                                      param_distributions = random_grid,
                                      n_iter = 100, cv = 3, verbose=2,
                                      random_state=10, n_jobs = 2)
RFC_RandomSearch.fit(XTrainRFAvH,YTrainRFAvH)

# Look at the Tuned "Best" Parameters
RFC_RandomSearch.best_params_
RFC_RandomSearch.best_score_
RFC_RandomSearch.best_estimator_.feature_importances_

# Fit using the best parameters
# Look at the Feature Importance
FeatImp = RFC_RandomSearch.best_estimator_.feature_importances_
NZInds = np.nonzero(FeatImp)
num_NZInds = len(NZInds[0])
Keep_NZVals = [x for x in FeatImp[NZInds[0]] if 
               (abs(x) >= np.mean(FeatImp[NZInds[0]]) 
               + 4*np.std(FeatImp[NZInds[0]]))]
ThreshVal = np.mean(FeatImp[NZInds[0]]) + 2*np.std(FeatImp[NZInds[0]])
Keep_NZInds = np.nonzero(abs(FeatImp[NZInds[0]]) >= ThreshVal)
Final_NZInds = NZInds[0][Keep_NZInds]



Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestRFAvH)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestRFAvH)

scikitplot.metrics.plot_roc(YTestRFAvH,Pred2_S2, title = 'ADvHC RF')
scikitplot.metrics.plot_confusion_matrix(YTestRFAvH,Pred1_S2)

accuracy = (Pred1_S2 == YTestRFAvH).mean() * 100.
print("FDA classification accuracy : %g%%" % accuracy) #57.1429%


# FDA
# Recover Classes Using Fisher's Linear Discriminant Analysis with SVD
[XTrainFDAvH,YTrainFDAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainAvH,YTrainAvH)
[XTestFDAvH,YTestFDAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestAvH,YTestAvH)

FDMod = LinearDiscriminantAnalysis(tol = 1e-4,solver = 'svd')
FDFit = FDMod.fit(XTrainFDAvH,YTrainFDAvH)

FIvec_FD_AvH = FDMod.coef_

FDTestPredADvsHC = FDFit.predict_proba(XTestFDAvH)
FDTestPred2ADvsHC = FDFit.predict(XTestFDAvH)
scikitplot.metrics.plot_roc(YTestFDAvH,FDTestPredADvsHC,title = 'FLD')
scikitplot.metrics.plot_confusion_matrix(YTestFDAvH,FDTestPred2ADvsHC)

accuracy = (FDTestPred2ADvsHC == YTestFDAvH).mean() * 100.
print("FDA classification accuracy : %g%%" % accuracy) 

FeatImp_RVC = abs(DVals)

feat_imp_RLR = feat_imp_RLR.T
feat_imp_RLR = feat_imp_RLR[:,0]

FeatMatrix = np.stack([FeatImp_RVC, feat_imp_RLR, FeatImp], axis = 1)
FeatCorr = np.corrcoef(FeatMatrix.T)
np.triu(FeatCorr)

plt.plot(FeatImp_RVC, FeatImp, 'o')

from scipy import stats

FeatImp_RVC_reshape = np.reshape(FeatImp_RVC,[19,16])
FeatImp_RVC_mean = np.mean(FeatImp_RVC_reshape, axis=0)
FeatImp_RVC_std = np.std(FeatImp_RVC_reshape, axis=0)
conf_int = stats.norm.interval(0.95, loc=FeatImp_RVC_mean, scale=FeatImp_RVC_std)
Freq_values = np.linspace(0,30,16)
plt.plot(Freq_values, FeatImp_RVC_mean, 'o')
plt.title("RVC Feature Importance by Channel")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean Feature Importance")
plt.show()



FeatImp_RF_reshape = np.reshape(FeatImp,[19,16])
FeatImp_RF_mean = np.mean(FeatImp_RF_reshape, axis=0)
FeatImp_RF_std = np.std(FeatImp_RF_reshape, axis=0)
conf_int = stats.norm.interval(0.95, loc=FeatImp_RF_mean, scale=FeatImp_RF_std)

Freq_values = np.linspace(0,30,16)
plt.plot(Freq_values,FeatImp_RF_mean, 'o')
plt.title("RF Feature Importance by Channel")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean Feature Importance")
plt.show()

