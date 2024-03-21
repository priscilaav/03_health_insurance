import math
import numpy  as np
import pandas as pd
import random
import sweetviz as sv
import seaborn as sns
import pickle
import warnings
import inflection
import seaborn as sns
import scikitplot as skplt

from scipy                           import stats  as ss
from boruta                          import BorutaPy
from matplotlib                      import pyplot as plt
from IPython.display                 import Image
from IPython.core.display            import HTML
from category_encoders               import TargetEncoder
from scipy.stats                     import shapiro
from scipy.stats                     import randint as sp_randint
from sklearn                         import preprocessing as pp
from sklearn                         import model_selection as ms
from sklearn                         import ensemble as en
from sklearn                         import neighbors as nh
from sklearn                         import linear_model    as lm
from sklearn.model_selection         import GridSearchCV
from sklearn.model_selection         import train_test_split
from sklearn.preprocessing           import RobustScaler, MinMaxScaler
from sklearn.ensemble                import RandomForestRegressor
from sklearn.ensemble                import AdaBoostClassifier
from sklearn.model_selection         import cross_val_score
from sklearn.tree                    import DecisionTreeClassifier
from sklearn                         import preprocessing as pp
from sklearn                         import model_selection as ms
from sklearn                         import ensemble as en
from sklearn                         import neighbors as nh
from sklearn                         import linear_model    as lm