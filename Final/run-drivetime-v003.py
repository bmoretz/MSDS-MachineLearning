# DriveTime Sedans (Python)

# Python v3.x. program revised by Thomas W. Milller (2018/08/23)

# Scikit Learn documentation for this study:
# http://scikit-learn.org/stable/modules/model_evaluation.html 
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.model_selection.KFold.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.LinearRegression.html
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.Ridge.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.Lasso.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.ElasticNet.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.metrics.r2_score.html
# sklearn.linear_model.LogisticRegression.html

# Categorical variable encoding
#   http://pbpython.com/categorical-encoding.html

# Textbook reference materials:
# 
# Geron, A. 2017. Hands-On Machine Learning with Scikit-Learn

# and TensorFlow. Sebastopal, Calif.: O'Reilly. Sample code from the
# book is available on GitHub at https://github.com/ageron/handson-ml
# 
# Izenman, A. J. 2008. Modern Multivariate Statistical Techniques: 
# Regression, Classification, and Manifold Learning. New York: Springer.
# [ISBN-13: 978-0-387-78188-4] Available from Springer collection at:
# http://link.springer.com.turing.library.northwestern.edu/
#
# Müller, A. C. and Guido, S. 2017. Introduction to Machine Learning with 
# Python: A Guide for Data Scientists. Sebastopol, Calif.: O’Reilly. 
# [ISBN-13: 978-1449369415] Code examples at 
# https://github.com/amueller/introduction_to_ml_with_python

# To obtain a listing of the results of this program, 
# locate yourself in the working direcotry and
# execute the following command in a terminal or commands window
# python run-drivetime-v003.py > list-drivetime-v003.txt

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import os
import time

# modeling routines from Scikit Learn packages
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# obtain precision, recall, F1, and support metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# specify the set of classifiers being evaluated
from sklearn.ensemble import RandomForestClassifier

# Pretty print a confusion matrixes rows actual, columns predicted
# Adapted from: https://gist.github.com/zachguo/10296432
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, 
             hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 9999

DATA_PATH = 'C:\Projects\MSDS-MachineLearning\Final'

print()
print('-------------------------------')
print('    Initial Data Prep       ')
print('-------------------------------')

data_file = os.path.join( DATA_PATH, 'drivetime-data-v001.csv')
# Data described in Appendix C.2 DriveTime Sedans of
# Miller, T. W. 2015. Modeling Techniques in Predictive Analytics 
# with Python and R: A Guide to Data Science. Upper Saddle River, N.J.: 
# Pearson. [ISBN-13: 978-0-13-389206-2]
print('\n----------------------------------------------')
print('DataFrame drivetime_input')
# Read data from comma-delimited text file and create Pandas DataFrame
drivetime_input = pd.read_csv(data_file)

# Check the initial Pandas DataFrame object
print('\nGeneral description of the drivetime_input DataFrame:')
print(drivetime_input.info())
print('\ndrivetime_input DataFrame (first and last five rows):')
print(drivetime_input.head())
print()
print(drivetime_input.tail())

# Partition the input DataFrame 
drivetime_build = drivetime_input.loc[drivetime_input['data_set'] == 'BUILD']
drivetime_dev = drivetime_input.loc[drivetime_input['data_set'] == 'DEV']
drivetime_holdout = drivetime_input.loc[drivetime_input['data_set'] == 'HOLDOUT']

print('\n----------------------------------------------')
print('DataFrame drivetime_build')
# Check the Pandas DataFrame object drivetime_build
print('\nGeneral description of the drivetime_build DataFrame:')
print(drivetime_build.info())
print(drivetime_build.describe(include = 'all'))
print('\ndrivetime_build DataFrame (first and last five rows):')
print(drivetime_build.head())
print()
print(drivetime_build.tail())

print('\n----------------------------------------------')
print('DataFrame drivetime_dev')
# Check the Pandas DataFrame object drivetime_dev
print('\nGeneral description of the drivetime_dev DataFrame:')
print(drivetime_dev.info())
print(drivetime_dev.describe(include = 'all'))
print('\ndrivetime_dev DataFrame (first and last five rows):')
print(drivetime_dev.head())
print()
print(drivetime_dev.tail())

print('\n----------------------------------------------')
print('DataFrame drivetime_holdout')
# Check the Pandas DataFrame object drivetime_holdout
print('\nGeneral description of the drivetime_holdout DataFrame:')
print(drivetime_holdout.info())
print(drivetime_holdout.describe(include = 'all'))
print('\ndrivetime_holdout DataFrame (first and last five rows):')
print(drivetime_holdout.head())
print()
print(drivetime_holdout.tail())

print()
print('-------------------------------')
print('     Study 1 Data Prep         ')
print('-------------------------------')
# Use drivetime_build from the DriveTime case to evaluate
# regression modeling methods within a cross-validation design.

# Set up preliminary data for data for evaluating alternative models 
# The first column is the total_cost response
# the remaining columns are the explanatory variables
prelim_model_data = \
    drivetime_build.filter(['total_cost', 'mileage', 
                            'vehicle_age', 'domestic_import', 'vehicle_type', 
                            'color_set'], 
                          axis = 1)

# Convert convert categorical variables (strings) to sets of binary variables
obj_df = prelim_model_data.select_dtypes(include=['object']).copy()
obj_df.head()
add_binary_df = pd.get_dummies(obj_df, 
                               columns=['domestic_import', 'vehicle_type',
                                        'color_set'], 
                               prefix=["mfg", "body", "color"])

# Merge binary indicator variables with other variables in data frame 
prelim_model_data = pd.concat([prelim_model_data, add_binary_df], axis=1) 


# Check the Pandas DataFrame object prelim_model_data
# Verify coding of binary variable import
print('\nGeneral description of the prelim_model_data DataFrame:')
print(prelim_model_data.info())
print('\nprelim_model_data DataFrame (first and last five rows):')
print(prelim_model_data.head())
print()
print(prelim_model_data.tail())

# Drop unnecessary variables
model_data = prelim_model_data.drop(['domestic_import',
                                     'vehicle_type',
                                     'color_set',
                                     'mfg_Import',
                                     'body_LUXURY',
                                     'color_WHITE'], 1)

# Check the Pandas DataFrame object model_data
# These are the data going into the multi-fold cross-validation
print('\nGeneral description of the model_data DataFrame:')
print(model_data.info())
print('\model_data DataFrame (first and last five rows):')
print(model_data.head())
print()
print(model_data.tail())

print('\nDimensions of model_data:', model_data.shape)

print()
print('-------------------------------')
print('     Study 1 Modeling          ')
print('-------------------------------')
# Specify the set of regression models being evaluated
# we set normalize=False because we have standardized
# the model input data outside of the modeling method calls
names = ['Linear_Regression', 
         'Ridge_Regression', 
         'Lasso_Regression', 
         'Random_Forest_Regression'] 

regressors = [LinearRegression(fit_intercept = True, 
                               normalize = True), 
              Ridge(alpha = 1, solver = 'cholesky', 
                    fit_intercept = True, 
                    normalize = True, 
                    random_state = RANDOM_SEED),               
              Lasso(alpha = 0.1, max_iter=1000, tol=0.0001, 
                    fit_intercept = True, 
                    normalize = True,
                    random_state = RANDOM_SEED),                                                                    
              RandomForestRegressor(n_estimators = 100, 
                                    criterion = 'mse', 
                                    max_depth = None, 
                                    min_samples_split = 200, 
                                    min_samples_leaf = 50, 
                                    min_weight_fraction_leaf = 0.0, 
                                    max_features = 14, 
                                    max_leaf_nodes = None, 
                                    min_impurity_split = 0.01, 
                                    bootstrap = True, 
                                    oob_score = False, 
                                    n_jobs=1, 
                                    random_state = RANDOM_SEED, 
                                    verbose = 0, 
                                    warm_start = False)]

# --------------------------------------------------------
# Specify the k-fold cross-validation design
from sklearn.model_selection import KFold

# Cross-validation set-up
N_FOLDS = 5

# Set up numpy arrays for storing RMSE and R-squared results
cv_rmse_results = np.zeros((N_FOLDS, len(names)))
cv_r2_results = np.zeros((N_FOLDS, len(names)))

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)
# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
#   the structure of modeling data for this study has the
#   response variable coming first and explanatory variables later          
#   so 1:model_data.shape[1] slices for explanatory variables
#   and 0 is the index for the response variable    
    X_train = model_data.iloc[train_index, 1:model_data.shape[1]]
    X_test = model_data.iloc[test_index, 1:model_data.shape[1]]
    y_train = model_data.iloc[train_index, 0]
    y_test = model_data.iloc[test_index, 0]  
    print('Shape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

    index_for_method = 0  # initialize
    for name, reg_model in zip(names, regressors):
        print('\nRegression model evaluation for:', name)
        print('  Scikit Learn method:', reg_model)
        reg_model.fit(X_train, y_train)  # fit on the train set for this fold

        # evaluate on the test set for this fold
        y_test_predict = reg_model.predict(X_test)
        fold_method_r2_result = r2_score(y_test, y_test_predict)
        print('Coefficient of determination (R-squared):',
              fold_method_r2_result)
        cv_r2_results[index_for_fold, index_for_method] = fold_method_r2_result
        fold_method_rmse_result = \
            np.power(mean_squared_error(y_test, y_test_predict), 0.5)
        print(reg_model.get_params(deep=True))
        print('Root mean-squared error:', fold_method_rmse_result)
        cv_rmse_results[index_for_fold, 
                        index_for_method] = fold_method_rmse_result
        index_for_method += 1
  
    index_for_fold += 1

cv_r2_results_df = pd.DataFrame(cv_r2_results)
cv_r2_results_df.columns = names

cv_rmse_results_df = pd.DataFrame(cv_rmse_results)
cv_rmse_results_df.columns = names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                        R2', sep = '')     
print(cv_r2_results_df.mean()) 

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                        RMSE', sep = '')     
print(cv_rmse_results_df.mean())


print()
print('-------------------------------')
print(' Study 2. 3, and 4 Data Prep   ')
print('-------------------------------')
# Prepare holdout data for use with a selected holdout
prelim_holdout_data = \
    drivetime_holdout.filter(['total_cost', 'mileage', 
                            'vehicle_age', 'domestic_import', 'vehicle_type',
                            'color_set'], 
                          axis = 1)

print('Preliminary holdout data descriptive statistics:')
print(prelim_holdout_data.describe())

# Convert convert categorical variables (strings) to sets of binary variables
obj_df = prelim_holdout_data.select_dtypes(include=['object']).copy()
obj_df.head()
add_binary_df = pd.get_dummies(obj_df, 
                               columns=['domestic_import', 'vehicle_type',
                                        'color_set'], 
                               prefix=["mfg", "body", "color"])

# Merge binary indicator variables with other variables in data frame 
prelim_holdout_data = pd.concat([prelim_holdout_data, add_binary_df], axis=1)
 
# Check the Pandas DataFrame object prelim_holdout_data
# Verify coding of binary variables
print('\nGeneral description of the prelim_holdout_data DataFrame:')
print(prelim_holdout_data.info())
print('\nprelim_holdout_data DataFrame (first and last five rows):')
print(prelim_holdout_data.head())
print()
print(prelim_holdout_data.tail())

# Drop unnecessary variables
holdout_data = prelim_holdout_data.drop(['domestic_import',
                                     'vehicle_type',
                                     'color_set',
                                     'mfg_Import',
                                     'body_LUXURY',
                                     'color_WHITE'], 1)

# Check the Pandas DataFrame object holdout_data
# These are the data going into the multi-fold cross-validation
print('\nGeneral description of the holdout_data DataFrame:')
print(prelim_holdout_data.info())
print('\holdout_data DataFrame (first and last five rows):')
print(holdout_data.head())
print()
print(holdout_data.tail())

print('\nDimensions of holdout_data:', holdout_data.shape)

print()
print('-------------------------------')
print('      Study 2 Modeling         ')
print('-------------------------------')
# Fit regression model to all the model_data from build data frame
X_train = model_data.iloc[:, 1:model_data.shape[1]]
y_train = model_data.iloc[:, 0]

X_holdout = holdout_data.iloc[:, 1:holdout_data.shape[1]]
y_holdout = holdout_data.iloc[:, 0]

print('\nShape of input data for this phase of the modeling work:',
          '\nData Set: (Observations, Variables)')
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)

print('X_holdout:',X_holdout.shape)
print('y_holdout:',y_holdout.shape)

reg_model_2 = LinearRegression(fit_intercept = True, normalize = False)
reg_model_2.fit(X_train, y_train)

print('\n----------------------------------------------')
# set up DataFrame for reporting results
var_name = [
    'intercept',    
    'mileage',
    'vehicle_age',
    'mfg_Domestic',
    'body_ECONOMY',
    'body_FAMILY.LARGE',
    'body_FAMILY.MEDIUM',
    'body_FAMILY.SMALL',
    'color_BLACK', 
    'color_BLUE',
    'color_GOLD', 
    'color_GREEN', 
    'color_PURPLE', 
    'color_RED',
    'color_SILVER']
var_description = [
    'Intercept',    
    'Mileage',
    'Vehicle Age (in years)',
    'Domestic Manufacturer',
    'Body Type = Economy',
    'Body Type = Large Family',
    'Body Type = Medium Family',
    'Body Type = Small Family',
    'Color = BLACK', 
    'Color = BLUE',
    'Color = GOLD', 
    'Color = GREEN', 
    'Color = PURPLE', 
    'Color = RED',
    'Color = SILVER']

coefficient = np.hstack((np.array([reg_model_2.intercept_]), 
                         reg_model_2.coef_))

reg_model_2_results = pd.DataFrame({'name': var_name,
                                    'description': var_description,
                                    'coefficient': coefficient})
print('Regression Model Results (fitted coefficients)')
print(reg_model_2_results)  

print('\nRoot mean-square error in training (build) data set: ',
      np.round_(np.power(mean_squared_error(y_train, 
                                           reg_model_2.predict(X_train)), 
                                           0.5), decimals = 2))
    
print('R-squared in training (build) data set: ',
      np.round_(np.power(r2_score(y_train, 
                                  reg_model_2.predict(X_train)), 
                                  0.5), decimals = 6))    
    
print('\nRoot mean-square error in holdout data set: ',
      np.round_(np.power(mean_squared_error(y_holdout, 
                                           reg_model_2.predict(X_holdout)), 
                                           0.5), decimals = 2))
    
print('R-squared in holdout data set: ',
      np.round_(np.power(r2_score(y_holdout, 
                                  reg_model_2.predict(X_holdout)), 
                                  0.5), decimals = 6))
        
    
print()
print('-------------------------------')
print('      Study 3 Modeling         ')
print('-------------------------------')
# Fit regression model to all the model_data from build data frame
# using standard scores for the explanatory variable columns... along axis 0
scaler = StandardScaler()
print(scaler.fit(model_data.iloc[:, 1:model_data.shape[1]]))
# show standardization constants being employed
print(scaler.mean_)
print(scaler.scale_)

model_data.head()

# the model data will be standardized form of preliminary model data
X_train = scaler.fit_transform(model_data.iloc[:, 1:model_data.shape[1]])
y_train = model_data.iloc[:, 0]  # keep response in original units

# Must standardize the holdout data as well
X_holdout = scaler.fit_transform(holdout_data.iloc[:, 1:holdout_data.shape[1]])
y_holdout = holdout_data.iloc[:, 0]

print('\nShape of input data for this phase of the modeling work:',
          '\nData Set: (Observations, Variables)')
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)

print('X_holdout:',X_holdout.shape)
print('y_holdout:',y_holdout.shape)


reg_model_3 = LinearRegression(fit_intercept = True, normalize = True)
reg_model_3.fit(X_train, y_train)  
print('\n----------------------------------------------')

# set up DataFrame for reporting results
var_name = [
    'intercept',    
    'mileage',
    'vehicle_age',
    'mfg_Domestic',
    'body_ECONOMY',
    'body_FAMILY.LARGE',
    'body_FAMILY.MEDIUM',
    'body_FAMILY.SMALL',
    'color_BLACK', 
    'color_BLUE',
    'color_GOLD', 
    'color_GREEN', 
    'color_PURPLE', 
    'color_RED',
    'color_SILVER']
var_description = [
    'Intercept',    
    'Mileage',
    'Vehicle Age (in years)',
    'Domestic Manufacturer',
    'Body Type = Economy',
    'Body Type = Large Family',
    'Body Type = Medium Family',
    'Body Type = Small Family',
    'Color = BLACK', 
    'Color = BLUE',
    'Color = GOLD', 
    'Color = GREEN', 
    'Color = PURPLE', 
    'Color = RED',
    'Color = SILVER']


coefficient = np.hstack((np.array([reg_model_3.intercept_]), reg_model_3.coef_))

reg_model_3_results = pd.DataFrame({'name': var_name,
                                    'description': var_description,
                                    'coefficient': coefficient})
print('Regression Model Results (fitted coefficients)')
print(reg_model_3_results)  

print('\nRoot mean-square error in training (build) data set: ',
      np.round_(np.power(mean_squared_error(y_train, 
                                           reg_model_3.predict(X_train)), 
                                           0.5), decimals = 2))
    
print('R-squared in training (build) data set: ',
      np.round_(np.power(r2_score(y_train, 
                                  reg_model_3.predict(X_train)), 
                                  0.5), decimals = 6))    
    
print('\nRoot mean-square error in holdout data set: ',
      np.round_(np.power(mean_squared_error(y_holdout, 
                                           reg_model_3.predict(X_holdout)), 
                                           0.5), decimals = 2))
    
print('R-squared in holdout data set: ',
      np.round_(np.power(r2_score(y_holdout, 
                                  reg_model_3.predict(X_holdout)), 
                                  0.5), decimals = 6))    
        
print()
print('-------------------------------')
print('      Study 4 Modeling         ')
print('-------------------------------')  
# Use same data as Study 3 
print('\nShape of input data for this phase of the modeling work:',
          '\nData Set: (Observations, Variables)')
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)

print('X_holdout:',X_holdout.shape)
print('y_holdout:',y_holdout.shape) 

reg_model_4 = RandomForestRegressor(n_estimators = 500, 
                                     criterion = 'mse', 
                                     max_depth = None, 
                                     min_samples_split = 200, 
                                     min_samples_leaf = 50, 
                                     min_weight_fraction_leaf = 0.0, 
                                     max_features = 14, 
                                     max_leaf_nodes = None, 
                                     min_impurity_split = 0.01, 
                                     bootstrap = True, 
                                     oob_score = False, 
                                     n_jobs=1, 
                                     random_state = RANDOM_SEED, 
                                     verbose = 0, 
                                     warm_start = False)

reg_model_4.fit(X_train, y_train)  
print('\n---------------------------------------------------------------------------')
print('Random Forests Regression Model Explanatory Variable Importance Results')
var_name = [   
    'mileage',
    'vehicle_age',
    'mfg_Domestic',
    'body_ECONOMY',
    'body_FAMILY.LARGE',
    'body_FAMILY.MEDIUM',
    'body_FAMILY.SMALL',
    'color_BLACK', 
    'color_BLUE',
    'color_GOLD', 
    'color_GREEN', 
    'color_PURPLE', 
    'color_RED',
    'color_SILVER']
var_description = [   
    'Mileage',
    'Vehicle Age (in years)',
    'Domestic Manufacturer',
    'Body Type = Economy',
    'Body Type = Large Family',
    'Body Type = Medium Family',
    'Body Type = Small Family',
    'Color = BLACK', 
    'Color = BLUE',
    'Color = GOLD', 
    'Color = GREEN', 
    'Color = PURPLE', 
    'Color = RED',
    'Color = SILVER']

var_importance = reg_model_4.feature_importances_
reg_model_4_results = pd.DataFrame({'name': var_name,
                                    'description': var_description,
                                    'importance': var_importance})
    
print(reg_model_4_results)    

print('\nRoot mean-square error in training (build) data set: ',
      np.round_(np.power(mean_squared_error(y_train, 
                                           reg_model_4.predict(X_train)), 
                                           0.5), decimals = 2))
    
print('R-squared in training (build) data set: ',
      np.round_(np.power(r2_score(y_train, 
                                  reg_model_4.predict(X_train)), 
                                  0.5), decimals = 6))    
    
print('\nRoot mean-square error in holdout data set: ',
      np.round_(np.power(mean_squared_error(y_holdout, 
                                           reg_model_4.predict(X_holdout)), 
                                           0.5), decimals = 2))
    
print('R-squared in holdout data set: ',
      np.round_(np.power(r2_score(y_holdout, 
                                  reg_model_4.predict(X_holdout)), 
                                  0.5), decimals = 6))    


print()
print('----------------------------------')
print(' Study 5, 6, 7, and 8 Data Prep   ')
print('----------------------------------')
# Create build_data DataFrame for training models
print('\nCreate build_data DataFrame')
# The first column is the binary response variable overage
# the remaining columns are the explanatory variables
# We work on the categorical variables first
prelim_build_data = \
    drivetime_build.filter(['overage',
                            'total_cost',
                            'domestic_import', 
                            'vehicle_type', 
                            'color_set', 
                            'makex',
                            'mileage',
                            'vehicle_age'], 
                          axis = 1)

# Convert convert categorical variables (strings) to sets of binary variables
obj_df = prelim_build_data.select_dtypes(include=['object']).copy()
obj_df.head()
add_binary_df = pd.get_dummies(obj_df, 
                               columns=['overage', 
                                        'domestic_import', 
                                        'vehicle_type',
                                        'color_set', 
                                        'makex'], 
                               prefix=['overage', 
                                       'mfg', 
                                       'body', 
                                       'color', 
                                       'make'])

# Merge binary indicator variables with other variables in data frame 
prelim_build_data = pd.concat([prelim_build_data, add_binary_df], axis=1)

print('Variables in prelim_build_data', prelim_build_data.columns)
 
# Check the Pandas DataFrame object prelim_build_data
# Verify coding of binary variables
print('\nCheck binary variable coding of the prelim_build_data DataFrame:')
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
    print(prelim_build_data)
    
# Work on continous explanatory variables
# Min-Max (that is, 0-to-1) scaling is used for continuous variables
min_max_scaler = MinMaxScaler()    
prelim_build_data[['total_cost', 'mileage', 'vehicle_age']] = \
    min_max_scaler.fit_transform(prelim_build_data[['total_cost', 
                                                    'mileage', 
                                                    'vehicle_age']])    
    
# Drop original categorical variables
build_data = prelim_build_data.drop(['overage', 
                                     'domestic_import', 
                                     'vehicle_type',
                                     'color_set', 
                                     'makex'], 1)
  
# Check the new Pandas DataFrame build_data
print('\nGeneral description of the build_data DataFrame:')
print(build_data.info())
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
    print(build_data)

print('\nDimensions of build_data:', build_data.shape)

print('\n----------------------------------------------')
print('\nCreate dev_data DataFrame')
# Create dev_data DataFrame for evaluating hyperparameter values
# The first column is the binary response variable overage
# the remaining columns are the explanatory variables
# We work on the categorical variables first
prelim_dev_data = \
    drivetime_dev.filter(['overage',
                          'total_cost',
                            'domestic_import', 
                            'vehicle_type', 
                            'color_set', 
                            'makex',
                            'mileage',
                            'vehicle_age'], 
                          axis = 1)

# Convert convert categorical variables (strings) to sets of binary variables
obj_df = prelim_dev_data.select_dtypes(include=['object']).copy()
obj_df.head()
add_binary_df = pd.get_dummies(obj_df, 
                               columns=['overage', 
                                        'domestic_import', 
                                        'vehicle_type',
                                        'color_set', 
                                        'makex'], 
                               prefix=['overage', 
                                       'mfg', 
                                       'body', 
                                       'color', 
                                       'make'])

# Merge binary indicator variables with other variables in data frame 
prelim_dev_data = pd.concat([prelim_dev_data, add_binary_df], axis=1)

print('Variables in prelim_dev_data', prelim_dev_data.columns)
 
# Check the Pandas DataFrame object prelim_dev_data
# Verify coding of binary variables
print('\nCheck binary variable coding of the prelim_dev_data DataFrame:')
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
    print(prelim_dev_data)
    
# Work on continous explanatory variables
# Min-Max (that is, 0-to-1) scaling is used for continuous variables
min_max_scaler = MinMaxScaler()    
prelim_dev_data[['total_cost', 'mileage', 'vehicle_age']] = \
    min_max_scaler.fit_transform(prelim_dev_data[['total_cost', 
                                                  'mileage', 
                                                  'vehicle_age']])    
    
# Drop original categorical variables
dev_data = prelim_dev_data.drop(['overage', 
                                     'domestic_import', 
                                     'vehicle_type',
                                     'color_set', 
                                     'makex'], 1)
  
# Check the new Pandas DataFrame dev_data
print('\nGeneral description of the dev_data DataFrame:')
print(dev_data.info())
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
    print(dev_data)

print('\nDimensions of dev_data:', dev_data.shape)

print('\n----------------------------------------------')
print('\nCreate final_test_data DataFrame')
# Create final_test_data DataFrame for testing models
# The first column is the binary response variable overage
# the remaining columns are the explanatory variables
# We work on the categorical variables first
prelim_final_test_data = \
    drivetime_holdout.filter(['overage', 
                              'total_cost',
                            'domestic_import', 
                            'vehicle_type', 
                            'color_set', 
                            'makex',
                            'mileage',
                            'vehicle_age'], 
                          axis = 1)

# Convert convert categorical variables (strings) to sets of binary variables
obj_df = prelim_final_test_data.select_dtypes(include=['object']).copy()
obj_df.head()
add_binary_df = pd.get_dummies(obj_df, 
                               columns=['overage', 
                                        'domestic_import', 
                                        'vehicle_type',
                                        'color_set', 
                                        'makex'], 
                               prefix=['overage', 
                                       'mfg', 
                                       'body', 
                                       'color', 
                                       'make'])

# Merge binary indicator variables with other variables in data frame 
prelim_final_test_data = pd.concat([prelim_final_test_data, add_binary_df], 
                                   axis=1)

print('Variables in prelim_final_test_data', prelim_final_test_data.columns)
 
# Check the Pandas DataFrame object prelim_final_test_data
# Verify coding of binary variables
print('\nCheck binary coding of the prelim_final_test_data DataFrame:')
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
    print(prelim_final_test_data)
    
# Work on continous explanatory variables
# Min-Max (that is, 0-to-1) scaling is used for continuous variables
min_max_scaler = MinMaxScaler()    
prelim_final_test_data[['total_cost', 'mileage', 'vehicle_age']] = \
    min_max_scaler.fit_transform(prelim_final_test_data[['total_cost',
                                                         'mileage', 
                                                         'vehicle_age']])    
    
# Drop original categorical variables
final_test_data = prelim_final_test_data.drop(['overage', 
                                     'domestic_import', 
                                     'vehicle_type',
                                     'color_set', 
                                     'makex'], 1)
  
# Check the new Pandas DataFrame final_test_data
print('\nGeneral description of the final_test_data DataFrame:')
print(final_test_data.info())
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
    print(final_test_data)

print('\nDimensions of final_test_data:', final_test_data.shape)

print()
print('-------------------------------')
print('      Study 5 Modeling         ')
print('-------------------------------')

# Set up input and output variables within training and test sets 
# from buld_data and final_test_data
X_build = build_data.filter(['total_cost', 
       'mileage', 'vehicle_age', 'mfg_Domestic',
       'mfg_Import', 'body_ECONOMY', 'body_FAMILY.LARGE', 'body_FAMILY.MEDIUM',
       'body_FAMILY.SMALL', 'body_LUXURY', 'color_BLACK', 'color_BLUE',
       'color_GOLD', 'color_GREEN', 'color_PURPLE', 'color_RED',
       'color_SILVER', 'color_WHITE', 'make_BUICK', 'make_CADILLAC',
       'make_CHEVROLET', 'make_CHRYSLER', 'make_DAEWOO', 'make_DODGE',
       'make_FORD', 'make_GEO', 'make_HONDA', 'make_HYUNDAI', 'make_KIA',
       'make_MAZDA', 'make_MERCURY', 'make_MITSUBISHI', 'make_NISSAN',
       'make_OLDSMOBILE', 'make_OTHER', 'make_PLYMOUTH', 'make_PONTIAC',
       'make_TOYOTA'], axis = 1)
  
y_build = build_data.filter(['overage_NO', 'overage_YES'])

X_final_test = final_test_data.filter(['total_cost',
       'mileage', 'vehicle_age', 'mfg_Domestic',
       'mfg_Import', 'body_ECONOMY', 'body_FAMILY.LARGE', 'body_FAMILY.MEDIUM',
       'body_FAMILY.SMALL', 'body_LUXURY', 'color_BLACK', 'color_BLUE',
       'color_GOLD', 'color_GREEN', 'color_PURPLE', 'color_RED',
       'color_SILVER', 'color_WHITE', 'make_BUICK', 'make_CADILLAC',
       'make_CHEVROLET', 'make_CHRYSLER', 'make_DAEWOO', 'make_DODGE',
       'make_FORD', 'make_GEO', 'make_HONDA', 'make_HYUNDAI', 'make_KIA',
       'make_MAZDA', 'make_MERCURY', 'make_MITSUBISHI', 'make_NISSAN',
       'make_OLDSMOBILE', 'make_OTHER', 'make_PLYMOUTH', 'make_PONTIAC',
       'make_TOYOTA'], axis = 1)

y_final_test = final_test_data.filter(['overage_NO', 'overage_YES'])

# check on shape of the training and test arrays
print('\nShape of X_build:', X_build.shape)
print('Shape of y_build:', y_build.shape)
print('Shape of X_final_test:', X_final_test.shape)
print('Shape of y_final_test:', y_final_test.shape)

# --------------------------------------------------------
# Assess elapsed time associated with the model fitting 
# ignoring input/output processing for the most part
start_time = time.clock()  # wall-clock time at beginning of modeling work

# define Study 5 classifier method as random forests 
clf = RandomForestClassifier(n_estimators=100, 
            criterion='gini', 
            max_depth=None, 
            min_samples_split=20, 
            min_samples_leaf=5, 
            min_weight_fraction_leaf=0.0, 
            max_features=38,
            max_leaf_nodes=None, 
            bootstrap=True, 
            oob_score=False, 
            n_jobs=1, 
            random_state=RANDOM_SEED, 
            verbose=0, 
            warm_start=False, 
            class_weight=None) 

# Fit classifier to the data and evaluate its performance
# in terms of precision, recall, and F1-score

clf.fit(X_build, y_build) 
end_time = time.clock()  # wall-clock time at end of modeling work
runtime = end_time - start_time  # seconds of wall-clock time 
print('\nRandom forest classification elapsed time (seconds):', 
      np.round(runtime, decimals = 3))

print('\n---------------------------------------------------------------------------')
print('Random Forests Regression Model Explanatory Variable Importance Results')
var_name = ['total_cost',
       'mileage', 'vehicle_age', 'mfg_Domestic',
       'mfg_Import', 'body_ECONOMY', 'body_FAMILY.LARGE', 'body_FAMILY.MEDIUM',
       'body_FAMILY.SMALL', 'body_LUXURY', 'color_BLACK', 'color_BLUE',
       'color_GOLD', 'color_GREEN', 'color_PURPLE', 'color_RED',
       'color_SILVER', 'color_WHITE', 'make_BUICK', 'make_CADILLAC',
       'make_CHEVROLET', 'make_CHRYSLER', 'make_DAEWOO', 'make_DODGE',
       'make_FORD', 'make_GEO', 'make_HONDA', 'make_HYUNDAI', 'make_KIA',
       'make_MAZDA', 'make_MERCURY', 'make_MITSUBISHI', 'make_NISSAN',
       'make_OLDSMOBILE', 'make_OTHER', 'make_PLYMOUTH', 'make_PONTIAC',
       'make_TOYOTA']
var_description = ['total_cost',
       'mileage', 'vehicle_age', 'mfg_Domestic',
       'mfg_Import', 'body_ECONOMY', 'body_FAMILY.LARGE', 'body_FAMILY.MEDIUM',
       'body_FAMILY.SMALL', 'body_LUXURY', 'color_BLACK', 'color_BLUE',
       'color_GOLD', 'color_GREEN', 'color_PURPLE', 'color_RED',
       'color_SILVER', 'color_WHITE', 'make_BUICK', 'make_CADILLAC',
       'make_CHEVROLET', 'make_CHRYSLER', 'make_DAEWOO', 'make_DODGE',
       'make_FORD', 'make_GEO', 'make_HONDA', 'make_HYUNDAI', 'make_KIA',
       'make_MAZDA', 'make_MERCURY', 'make_MITSUBISHI', 'make_NISSAN',
       'make_OLDSMOBILE', 'make_OTHER', 'make_PLYMOUTH', 'make_PONTIAC',
       'make_TOYOTA']
var_importance = clf.feature_importances_
final_model_results = pd.DataFrame({'name': var_name,
                                    'description': var_description,
                                    'importance': var_importance})
    
print(final_model_results)     


# Report results of random forest classifier
overage_labels = ['NO', 'YES'] 
y_final_test_predict = clf.predict(X_final_test)

print('\nRandom Forest Confusion Matrix (rows actual, columns predicted)\n')
print_cm(confusion_matrix(y_final_test.values.argmax(axis=1), 
                 y_final_test_predict.argmax(axis=1)),
                 labels = overage_labels)

print('\nRandom Forest Predictive Accuracy: ',                
np.round(accuracy_score(y_final_test.values.argmax(axis=1), 
               y_final_test_predict.argmax(axis=1)), decimals = 3))

print('\nSummary of Random Forest Precision and Recall\n') 
             
print(classification_report(y_final_test, y_final_test_predict, 
                            target_names = overage_labels))

print('precision = proportion of returned results that are relevant')  
print('recall = proportion of relevant documents that are returned')   
print('f1-score = harmonic mean of precision and recall')

print()
print('-------------------------------')
print('      Study 6 Modeling         ')
print('-------------------------------') 

# tensorflow
import tensorflow as tf
from functools import partial

# Function to make output stable across runs
def reset_graph(seed = 1):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Activation function
def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha*z, z)

n_inputs = X_build.shape[1]
n_outputs = y_build.shape[1]

X_dev = dev_data.filter(['total_cost',
       'mileage', 'vehicle_age', 'mfg_Domestic',
       'mfg_Import', 'body_ECONOMY', 'body_FAMILY.LARGE', 'body_FAMILY.MEDIUM',
       'body_FAMILY.SMALL', 'body_LUXURY', 'color_BLACK', 'color_BLUE',
       'color_GOLD', 'color_GREEN', 'color_PURPLE', 'color_RED',
       'color_SILVER', 'color_WHITE', 'make_BUICK', 'make_CADILLAC',
       'make_CHEVROLET', 'make_CHRYSLER', 'make_DAEWOO', 'make_DODGE',
       'make_FORD', 'make_GEO', 'make_HONDA', 'make_HYUNDAI', 'make_KIA',
       'make_MAZDA', 'make_MERCURY', 'make_MITSUBISHI', 'make_NISSAN',
       'make_OLDSMOBILE', 'make_OTHER', 'make_PLYMOUTH', 'make_PONTIAC',
       'make_TOYOTA'], axis = 1)
    
# Scikit Learn for min-max scaling of the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(np.concatenate((X_dev, X_build, X_final_test), axis = 0))
X_dev = scaler.transform(X_dev)
X_build = scaler.transform(X_build)
X_final_test = scaler.transform(X_final_test)
    
y_dev = dev_data.filter(['overage_NO', 'overage_YES'])

# check training and dev/test arrays
print('\nShape of X_build:', X_build.shape)
print('First two X_build vehicles:')
print(X_build[0:1,:])
print('Shape of X_dev:', X_dev.shape)
print('First two X_dev vehicles:')
print(X_dev[0:1,:])
print('Shape of X_final_test:', X_final_test.shape)
print('First two X_final_test vehicles:')
print(X_final_test[0:1,:])
print('Shape of y_build:', y_build.shape)
print('Shape of y_dev:', y_dev.shape)
print('Shape of y_final_test:', y_final_test.shape)

# Set hyperparameter lists
n_hidden1_list = [2, 4, 8, 16]
n_hidden2_list = [2, 4, 8, 16]

batch_norm_momentum = 0.9
learning_rate = 0.01
n_epochs = 100

OUTPUT_PATH = os.path.join(DATA_PATH, 'output')

output_file = os.path.join(OUTPUT_PATH, "my_model_final.ckpt")

for n_hidden1 in n_hidden1_list:
    for n_hidden2 in n_hidden2_list:
        print()
        print('----------------------------------------------')
        print(n_hidden1, ' nodes in first hidden layer')
        print(n_hidden2, ' nodes in second hidden layer')

        reset_graph()

        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int64, shape=(None), name="y")
        training = tf.placeholder_with_default(False, shape=(), 
                                               name='training')

        with tf.name_scope("dnn"):
            he_init = tf.contrib.layers.variance_scaling_initializer()

            my_batch_norm_layer = partial(
                    tf.layers.batch_normalization,
                    training=training,
                    momentum=batch_norm_momentum)

            my_dense_layer = partial(
                    tf.layers.dense,
                    kernel_initializer=he_init)

            hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
            bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
            hidden2 = my_dense_layer(bn1, n_hidden2, name="hidden2")
            bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
            logits_before_bn = my_dense_layer(bn2, n_outputs, name="outputs")
            logits = my_batch_norm_layer(logits_before_bn)

        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, 
                logits=logits)
            loss = tf.reduce_mean(xentropy, name="loss")

        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            training_op = optimizer.minimize(loss)

        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
               
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()     
        
        with tf.Session() as sess:
            init.run()
            start_time = time.clock()  # wall-clock time at beginning 
            for epoch in range(n_epochs):

                sess.run(training_op, feed_dict={X: X_build, 
                                                 y: y_build.iloc[:,1]})
                
                if (epoch % 20 == 0) or (epoch == n_epochs -1):
                    acc_train = accuracy.eval(feed_dict={X: X_build, 
                                                         y: y_build.iloc[:,1]})
                    acc_test = accuracy.eval(feed_dict={X: X_dev, 
                                                        y: y_dev.iloc[:,1]})
                    print(epoch, "Build set accuracy:", acc_train, 
                              "Dev set accuracy:", acc_test)

            end_time = time.clock()  # wall-clock time at end of modeling work
            runtime = end_time - start_time  # seconds of wall-clock time 
            print('\nNeural network elapsed time (seconds):', 
                np.round(runtime, decimals = 3))                
                
            save_path = saver.save(sess, output_file)                

print()
print('-------------------------------')
print('      Study 7 Modeling         ')
print('-------------------------------') 

# Evaluate on final test
print('\n----------------------------------------------')
print('Selected neural network architecture:')
print(n_hidden1, ' nodes in first hidden layer')
print(n_hidden2, ' nodes in second hidden layer') 

print()
n_epochs = 500      
with tf.Session() as sess:
    init.run()
    start_time = time.clock()  # wall-clock time at beginning 
    
    for epoch in range(n_epochs):

        sess.run(training_op, feed_dict={X: X_build, 
                                         y: y_build.iloc[:,1]})
                
        if (epoch % 50 == 0) or (epoch == n_epochs -1):
            acc_train = accuracy.eval(feed_dict={X: X_build, 
                                                 y: y_build.iloc[:,1]})
            acc_final_test = accuracy.eval(feed_dict={X: X_final_test, 
                                          y: y_final_test.iloc[:,1]})
            print(epoch, "Build set accuracy:", acc_train, 
                         "Final test set accuracy:", acc_final_test)
            
            prediction = tf.argmax(logits, 1)
            y_final_test_pred = prediction.eval(feed_dict={X: X_final_test})

    end_time = time.clock()  # wall-clock time at end of modeling work 
    runtime = end_time - start_time  # seconds of wall-clock time 
    print('\nNeural network elapsed time (seconds):', 
        np.round(runtime, decimals = 3))  
    save_path = saver.save(sess, "./my_model_final.ckpt")

print('\nNeural Network Confusion Matrix (rows actual, columns predicted)\n')
print_cm(confusion_matrix(y_final_test.values.argmax(axis=1), 
                 y_final_test_pred),
                 labels = overage_labels)

print('\nNeural Network Accuracy: ',                
np.round(accuracy_score(y_final_test.values.argmax(axis=1), 
               y_final_test_pred), decimals = 3))
   
print()
print('-------------------------------')
print('      Study 8 Modeling         ')
print('-------------------------------') 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

print('\nShape of X_build:', X_build.shape)
print('Shape of y_build:', y_build.shape)
print('Shape of X_final_test:', X_final_test.shape)
print('Shape of y_final_test:', y_final_test.shape)

classifier = LogisticRegression()
classifier.fit(X_build, y_build.iloc[:,1])  
# evaluate on the test set for this fold
y_final_test_logit_predict_prob = classifier.predict_proba(X_final_test)
y_final_test_logit_predict = classifier.predict(X_final_test)

print('Area under ROC curve:', roc_auc_score(y_final_test.iloc[:,1], 
                                    y_final_test_logit_predict_prob[:,1]))

print('\nLogistic Regression Confusion Matrix',
      '(rows actual, columns predicted)\n')
print_cm(confusion_matrix(y_final_test.values.argmax(axis=1), 
                 y_final_test_logit_predict),
                 labels = overage_labels)

print('\nLogistic Regression Accuracy: ',                
np.round(accuracy_score(y_final_test.values.argmax(axis=1), 
               y_final_test_logit_predict), decimals = 3))
          
print()
print('RUN COMPLETE')