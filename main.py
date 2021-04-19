"""
Copyright (c) 2021, Heung Kit Leslie Chung
All rights reserved.
Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, make_scorer, classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier, XGBRegressor, plot_tree, plot_importance

class uplift_model:
    def __init__(self):
        pass
    
    def mu_training(self, X, t, y):
        '''
        This function takes in X, t and y and outputs trained models mu0
        and mu1. The data is first split into control and treatment groups
        based on t and each model is trained separately on their respective
        groups. 
        
        This will then give us mu0 -- model trained on control group, and 
        mu1 -- model trained on treatment group. Both models are trained to 
        predict the outcome.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Here are your model's featuress.
        t : pandas.core.series.Series
            Here is a boolean series for treatment.
        y : pandas.core.series.Series
            Here is a boolean series for outcome.

        Returns
        -------
        clf0 : xgboost.sklearn.XGBClassifier
            Our mu0 model -- model trained on control group.
        clf1 : xgboost.sklearn.XGBClassifier
            Our mu1 model -- model trained on treatment group.

        '''
        temp = X[:]
        temp['treatment'] = t
        temp['conversion'] = y
        
        train0 = temp[temp['treatment'] == 0]
        train1 = temp[temp['treatment'] == 1]
        
        # If binary outcome majority class exceeds 70% then do upsampling on minority class, else do nothing
        if max(temp['conversion'].value_counts(normalize=True)) > 0.7:
        
            # Resample control group
            df_majority = train0[train0.conversion==0]
            df_minority = train0[train0.conversion==1]
            df_minority_upsampled = resample(df_minority, 
                                             replace=True,     # sample with replacement
                                             n_samples=len(df_majority),    # to match majority class
                                             random_state=123) # reproducible results
            
            # Combine majority class with upsampled minority class
            df_upsampled = pd.concat([df_majority, df_minority_upsampled])
            
            # Resample treatment group
            df_majority2 = train1[train1.conversion==0]
            df_minority2 = train1[train1.conversion==1]
            df_minority2_upsampled = resample(df_minority2, 
                                             replace=True,     # sample with replacement
                                             n_samples=len(df_majority2),    # to match majority class
                                             random_state=123) # reproducible results
            
            # Combine majority class with upsampled minority class
            df_upsampled2 = pd.concat([df_majority2, df_minority2_upsampled])
            
            # Combine both upsampled sets together
            train = df_upsampled.append(df_upsampled2)
            
            # Resplit the control/treatment groups
            train0 = train[train['treatment'] == 0]
            train1 = train[train['treatment'] == 1]
            
        else:
            pass
        
        # Split control/treatment groups into train and validation sets
        X_train0, X_val0, y_train0, y_val0 = train_test_split(train0.iloc[:,:-2], train0['conversion'], test_size=0.2, random_state=0)
        X_train1, X_val1, y_train1, y_val1 = train_test_split(train1.iloc[:,:-2], train1['conversion'], test_size=0.2, random_state=0)
        
        # mu0
        clf0 = XGBClassifier(eta = 0.5, max_depth = 5, seed = 0, gamma = 2)
        clf0.fit(X_train0, y_train0)
        clf0_predictions = clf0.predict(X_val0)
        clf0_accuracy = accuracy_score(y_val0, clf0_predictions)
        clf0_f1 = f1_score(y_val0, clf0_predictions)
        clf0_score = clf0.predict_proba(X_val0)
        clf0_gini = (roc_auc_score(y_val0, clf0_score[:,1]) - 0.5) * 2
        print("mu0 XGBoost Classifier for control: [Accuracy: {:.4f}, F1-score: {:.4f}, Gini: {:.4f}]".format(clf0_accuracy, 
                                                                                                clf0_f1,
                                                                                                clf0_gini))
        
        # mu1
        clf1 = XGBClassifier(eta = 0.5, max_depth = 5, seed = 0, gamma = 2)
        clf1.fit(X_train1, y_train1)
        clf1_predictions = clf1.predict(X_val1)
        clf1_accuracy = accuracy_score(y_val1, clf1_predictions)
        clf1_f1 = f1_score(y_val1, clf1_predictions)
        clf1_score = clf1.predict_proba(X_val1)
        clf1_gini = (roc_auc_score(y_val1, clf1_score[:,1]) - 0.5) * 2
        print("mu1 XGBoost Classifier for treatment: [Accuracy: {:.4f}, F1-score: {:.4f}, Gini: {:.4f}]".format(clf1_accuracy, 
                                                                                                clf1_f1,
                                                                                                clf1_gini))
    
        return clf0, clf1

    def calculate_ite(self, X, t, y, clf0, clf1):
        '''
        This function takes in X, t, y and two classifier models, namely the
        mu0 and mu1, and outputs the two models' predictions and the
        individual treatment effect values. 
        
        ITE is calculated as follows:
            Control ITE = mu1(x) - control group observed outcome
            Treatment ITE = treatment group observed outcome - mu0(x)

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Here are your model's features.
        t : pandas.core.series.Series
            Here is a boolean series for treatment.
        y : pandas.core.series.Series
            Here is a boolean series for outcome.
        clf0 : xgboost.sklearn.XGBClassifier
            Our mu0 model -- model trained on control group.
        clf1 : xgboost.sklearn.XGBClassifier
            Our mu1 model -- model trained on treatment group.

        Returns
        -------
        mu0(x) : pandas.core.series.Series
            mu0's predictions.
        mu1(x) : pandas.core.series.Series
            mu1's predictions.
        ite : pandas.core.series.Series
            Individual treatment effects (ITEs).
        
        '''
        temp = X[:]
        temp['treatment'] = t
        temp['conversion'] = y
        
        # Using mu0 and mu1 to predict propensity scores
        temp['mu0'] = clf0.predict_proba(temp.iloc[:,:-2])[:,1]
        temp['mu1'] = clf1.predict_proba(temp.iloc[:,:-3])[:,1]
        
        # Control ITE = treatment group model prediction - control group observed outcome
        # Treatment ITE = treatment group observed outcome - control group model prediction
        temp['ite'] = [(temp.iloc[x]['mu1'] - temp.iloc[x]['conversion']) if temp.iloc[x]['treatment'] == 0 else (temp.iloc[x]['conversion'] - temp.iloc[x]['mu0']) for x  in range(0, len(temp))]
    
        return temp['mu0'], temp['mu1'], temp['ite']
    
    def calculate_propensity_diff(self, X, t, y, clf0, clf1):
        '''
        This function takes in X, t, y and two classifier models, namely the
        mu0 and mu1, and outputs the two models' predictions and the
        uplift which is taken as mu1 - mu0.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Here are your model's features.
        t : pandas.core.series.Series
            Here is a boolean series for treatment.
        y : pandas.core.series.Series
            Here is a boolean series for outcome.
        clf0 : xgboost.sklearn.XGBClassifier
            Our mu0 model -- model trained on control group.
        clf1 : xgboost.sklearn.XGBClassifier
            Our mu1 model -- model trained on treatment group.

        Returns
        -------
        mu0(x) : pandas.core.series.Series
            mu0's predictions.
        mu1(x) : pandas.core.series.Series
            mu1's predictions.
        uplift : pandas.core.series.Series
            mu1 - mu0.

        '''
        temp = X[:]
        temp['treatment'] = t
        temp['conversion'] = y
        
        # Using mu0 and mu1 to predict propensity scores
        temp['mu0'] = clf0.predict_proba(temp.iloc[:,:-2])[:,1]
        temp['mu1'] = clf1.predict_proba(temp.iloc[:,:-3])[:,1]
        
        # Difference between treatment propensity and control propensity gives
        # the uplift in tlearner
        temp['uplift'] = temp['mu1'] - temp['mu0']
        
        return temp['mu0'], temp['mu1'], temp['uplift']
    
    def tau_training(self, X, t, y, ite):
        '''
        This function takes in X, t, y and the ITE values and outputs trained 
        models tau0 and tau1.The data is first split into control and treatment 
        groups based on t and each model is trained separately on their 
        respective groups. 
        
        This will then give us tau0 -- model trained on control group, and 
        tau1 -- model trained on treatment group. Both models are trained to 
        predict the ITEs in respective groups.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Here are your model's featuress.
        t : pandas.core.series.Series
            Here is a boolean series for treatment.
        y : pandas.core.series.Series
            Here is a boolean series for outcome.
        ite : pandas.core.series.Series
            Individual treatment effects (ITEs).

        Returns
        -------
        reg0 : xgboost.sklearn.XGBRegressor
            Our tau0 model -- model trained on control group.
        reg1 : xgboost.sklearn.XGBRegressor
            Our tau1 model -- model trained on treatment group.

        '''
        temp = X[:]
        temp['treatment'] = t
        temp['conversion'] = y
        temp['ite'] = ite
        
        train0 = temp[temp['treatment'] == 0]
        train1 = temp[temp['treatment'] == 1]
        
        # Split control/treatment groups into train and validation sets
        X_train0, X_val0, y_train0, y_val0 = train_test_split(train0.iloc[:,:-3], train0['ite'], test_size=0.2, random_state=0)
        X_train1, X_val1, y_train1, y_val1 = train_test_split(train1.iloc[:,:-3], train1['ite'], test_size=0.2, random_state=0)
        
        # tau0
        reg0 = XGBRegressor(verbosity=0, random_state=0, max_depth=5)
        reg0.fit(X_train0, y_train0)
        reg0_predictions = reg0.predict(X_val0)
        reg0_mse = mean_squared_error(y_val0, reg0_predictions)
        reg0_r2 = r2_score(y_val0, reg0_predictions)
        print("tau0 XGBoost Regressor: [Mean Squared Error: {:.4f}, R2-score: {:.4f}]".format(reg0_mse, reg0_r2))
        
        # tau1
        reg1 = XGBRegressor(verbosity=0, random_state=0, max_depth=5)
        reg1.fit(X_train1, y_train1)
        reg1_predictions = reg1.predict(X_val1)
        reg1_mse = mean_squared_error(y_val1, reg1_predictions)
        reg1_r2 = r2_score(y_val1, reg1_predictions)
        print("tau1 XGBoost Regressor: [Mean Squared Error: {:.4f}, R2-score: {:.4f}]".format(reg1_mse, reg1_r2))
        
        return reg0, reg1
    
    def cross_predict_ite(self, X, t, y, reg0, reg1):
        '''
        This function takes in X, t, y and two regressor models, namely the
        tau0 and tau1, and outputs the two models' predicted
        individual treatment effect values.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Here are your model's featuress.
        t : pandas.core.series.Series
            Here is a boolean series for treatment.
        y : pandas.core.series.Series
            Here is a boolean series for outcome.
        reg0 : xgboost.sklearn.XGBRegressor
            Our tau0 model -- model trained on control group.
        reg1 : xgboost.sklearn.XGBRegressor
            Our tau1 model -- model trained on treatment group.

        Returns
        -------
        tau0(x) : pandas.core.series.Series
            tau0's predictions.
        tau1(x) : pandas.core.series.Series
            tau1's predictions.

        '''
        temp = X[:]
        temp['treatment'] = t
        temp['conversion'] = y
        
        # Useing tau0 and tau1 to predict ITEs
        temp['tau0_ite'] = reg0.predict(temp.iloc[:,:-2])
        temp['tau1_ite'] = reg1.predict(temp.iloc[:,:-3])
        
        return temp['tau0_ite'], temp['tau1_ite']
    
    def calc_uplift(self, X, t, y, tau0_ite, tau1_ite):
        '''
        This function takes in X, t, y and the two regressor models'
        predictions, and outputs the uplifts.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Here are your model's featuress.
        t : pandas.core.series.Series
            Here is a boolean series for treatment.
        y : pandas.core.series.Series
            Here is a boolean series for outcome.
        tau0_ite : pandas.core.series.Series
            tau0's predictions.
        tau1_ite : pandas.core.series.Series
            tau1's predictions.

        Returns
        -------
        uplift : pandas.core.series.Series
            Uplift derived from weighted sum of tau0_ite and tau1_ite.

        '''
        temp = X[:]
        temp['treatment'] = t
        temp['conversion'] = y
        temp['tau0_ite'] = tau0_ite
        temp['tau1_ite'] = tau1_ite
        
        # Calculate uplift
        temp['uplift'] = (0.5*temp['tau0_ite']) + (0.5*temp['tau1_ite'])
        
        return temp['uplift']
    
    def slearner_uplift(self, X, t, y):
        '''
        This functiion takes in X, t, y and outputs a table with uplift
        predictions and single classifier model.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Here are your model's featuress.
        t : pandas.core.series.Series
            Here is a boolean series for treatment.
        y : pandas.core.series.Series
            Here is a boolean series for outcome.

        Returns
        -------
        final : pandas.core.frame.DataFrame
            DESCRIPTION.
        clf0 : xgboost.sklearn.XGBClassifier
            Our mu0 model -- model trained on control group.

        '''
        temp = X[:]
        temp['treatment'] = t
        temp['conversion'] = y
        
        train = temp
        
        # If binary outcome majority class exceeds 70% then do upsampling on minority class, else do nothing
        if max(train['conversion'].value_counts(normalize=True)) > 0.7:
        
            # Resample control group
            df_majority = train[train.conversion==0]
            df_minority = train[train.conversion==1]
            df_minority_upsampled = resample(df_minority, 
                                             replace=True,     # sample with replacement
                                             n_samples=len(df_majority),    # to match majority class
                                             random_state=123) # reproducible results
            
            # Combine majority class with upsampled minority class
            train = pd.concat([df_majority, df_minority_upsampled])
            
        else:
            pass
        
        # Split control/treatment groups into train and validation sets
        X_train0, X_val0, y_train0, y_val0 = train_test_split(train.iloc[:,:-1], train['conversion'], test_size=0.2, random_state=0)
        
        # Train a regular propensity model for the conversion target column
        clf0 = XGBClassifier(eta = 0.5, max_depth = 5, seed = 0, gamma = 2)
        clf0.fit(X_train0, y_train0)
        clf0_predictions = clf0.predict(X_val0)
        clf0_accuracy = accuracy_score(y_val0, clf0_predictions)
        clf0_f1 = f1_score(y_val0, clf0_predictions)
        clf0_score = clf0.predict_proba(X_val0)
        clf0_gini = (roc_auc_score(y_val0, clf0_score[:,1]) - 0.5) * 2
        print("Conversion propensity XGBoost Classifier: [Accuracy: {:.4f}, F1-score: {:.4f}, Gini: {:.4f}]".format(clf0_accuracy, 
                                                                                                clf0_f1,
                                                                                                clf0_gini))
        
        # Scoring step
        temp = X[:]
        
        # First to score for control
        temp['treatment'] = 0
        temp['conversion'] = y
        # Using the propensity model to predict propensity scores for control
        temp['c_prop'] = clf0.predict_proba(temp.iloc[:,:-1])[:,1]
        
        # Then to score for treatment
        temp['treatment'] = 1
        # Using the propensity model to predict propensity scores for treatment
        temp['t_prop'] = clf0.predict_proba(temp.iloc[:,:-2])[:,1]
        
        # Reset treatment column
        temp['treatment'] = t
        
        # Difference between treatment propensity and control propensity gives
        # the uplift in slearner
        temp['uplift'] = temp['t_prop'] - temp['c_prop']
        
        # Prepare a final table for output
        final = temp
        
        return final, clf0
    
    def tlearner_uplift(self, X, t, y):
        '''
        This functiion takes in X, t, y and outputs a table with uplift
        predictions and 2 classifier models (mu0 and mu1).

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Here are your model's featuress.
        t : pandas.core.series.Series
            Here is a boolean series for treatment.
        y : pandas.core.series.Series
            Here is a boolean series for outcome.

        Returns
        -------
        final : pandas.core.frame.DataFrame
            DESCRIPTION.
        clf0 : xgboost.sklearn.XGBClassifier
            Our mu0 model -- model trained on control group.
        clf1 : xgboost.sklearn.XGBClassifier
            Our mu1 model -- model trained on treatment group.

        '''
        # Calling mu_training func
        clf0, clf1 = self.mu_training(X, t, y)
        
        # Calling calculate_propensity_diff func (this is the uplift from tlearner)
        mu0, mu1, uplift = self.calculate_propensity_diff(X, t, y, clf0, clf1)
        
        # Prepare a final table for output
        final = X[:]
        final['treatment'] = t
        final['conversion'] = y
        final['mu0'] = mu0
        final['mu1'] = mu1
        final['uplift'] = uplift
        
        return final, clf0, clf1
    
    def xlearner_uplift(self, X, t, y):
        '''
        This functiion takes in X, t, y and outputs a table with uplift
        predictions, 2 classifier models (mu0 and mu1) and 2 regressor models
        (tau0 and tau1).

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Here are your model's featuress.
        t : pandas.core.series.Series
            Here is a boolean series for treatment.
        y : pandas.core.series.Series
            Here is a boolean series for outcome.

        Returns
        -------
        final : pandas.core.frame.DataFrame
            DESCRIPTION.
        clf0 : xgboost.sklearn.XGBClassifier
            Our mu0 model -- model trained on control group.
        clf1 : xgboost.sklearn.XGBClassifier
            Our mu1 model -- model trained on treatment group.
        reg0 : xgboost.sklearn.XGBRegressor
            Our tau0 model -- model trained on control group.
        reg1 : xgboost.sklearn.XGBRegressor
            Our tau1 model -- model trained on treatment group.

        '''
        # Calling mu_training func
        clf0, clf1 = self.mu_training(X, t, y)
        
        # Calling calculate_ite func
        mu0, mu1, ite = self.calculate_ite(X, t, y, clf0, clf1)
        
        # Calling tau_training func
        reg0, reg1 = self.tau_training(X, t, y, ite)
        
        # Calling cross_predict_ite func
        tau0_ite, tau1_ite = self.cross_predict_ite(X, t, y, reg0, reg1)
        
        # Calling calc_uplift_func
        uplift = self.calc_uplift(X, t, y, tau0_ite, tau1_ite)
        
        # Prepare a final table for output
        final = X[:]
        final['treatment'] = t
        final['conversion'] = y
        final['mu0'] = mu0
        final['mu1'] = mu1
        final['tau0_ite'] = tau0_ite
        final['tau1_ite'] = tau1_ite
        final['uplift'] = uplift
        
        return final, clf0, clf1, reg0, reg1
