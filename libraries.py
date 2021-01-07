import pandas as pd
import numpy as np
import math
import seaborn as sns
import scipy.stats as ss
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import seaborn as sns

from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import shap
import argparse

from sklearn.metrics import roc_auc_score, roc_curve, auc, r2_score, mean_squared_error, accuracy_score
from itertools import cycle
from skopt import gp_minimize, Optimizer, space

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer

from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv

from bayes_opt import BayesianOptimization
import time
import gc
gc.collect()

pd.pandas.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

