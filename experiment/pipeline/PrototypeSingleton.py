from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, PowerTransformer, KBinsDiscretizer, \
    Binarizer, OneHotEncoder, OrdinalEncoder

from .utils import generate_domain_space

import pandas as pd

class PrototypeSingleton:
   __instance = None

   POOL = {
       "imputate": [None, SimpleImputer(), IterativeImputer()],
       "encode": [OneHotEncoder(), OrdinalEncoder()],
       "rebalance": [None, NearMiss(), SMOTE()],
       #"rebalance": [None, NearMiss(), CondensedNearestNeighbour(), SMOTE()],
       "normalizer": [None, StandardScaler(), PowerTransformer(), MinMaxScaler(), RobustScaler()],
       "discretize": [None, KBinsDiscretizer(), Binarizer()],
       "features": [None, PCA(), SelectKBest(), FeatureUnion([("pca", PCA()), ("selectkbest", SelectKBest())])]
   }

   PROTOTYPE = {}
   DOMAIN_SPACE = {}
   parts = []
   X = []
   y = []
   numerical_features = []
   categorical_features = []


   @staticmethod
   def getInstance():
      """ Static access method. """
      if PrototypeSingleton.__instance == None:
         PrototypeSingleton()
      return PrototypeSingleton.__instance

   def __init__(self):
      """ Virtually private constructor. """
      if PrototypeSingleton.__instance != None:
         raise Exception("This class is a singleton!")
      else:
         PrototypeSingleton.__instance = self

   def setPipeline(self, params):
       for param in params:
           self.parts.append(param)

       for part in self.parts:
           self.PROTOTYPE[part] = self.POOL[part]

       self.DOMAIN_SPACE = generate_domain_space(self.PROTOTYPE)

   def setDataset(self, X, y):
       self.X = pd.DataFrame(X)
       self.y = pd.DataFrame(y)
       self.numerical_features = self.X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
       self.categorical_features = self.X.select_dtypes(include=['object']).columns

   def discretizeFeatures(self):
       self.numerical_features = []
       self.categorical_features = self.X.columns

   def getFeatures(self):
       return self.numerical_features, self.categorical_features

   def getDomainSpace(self):
       return self.DOMAIN_SPACE

   def getPrototype(self):
       return self.PROTOTYPE

   def getParts(self):
       return self.parts