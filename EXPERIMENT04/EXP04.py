# -*- coding: utf-8 -*-
"""machinelearningb50EXp04.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dp1IOuSro6-e1UqrWFGu_iFLQ_RV0R3O
"""

import pandas as pd

from google.colab import files

upload=files.upload()

import io

df = pd.read_csv(io.BytesIO(upload['holdout.csv']))

df.head(5)

df.info();

df.describe()

df.isnull().sum()

df.isnull().sum().sum()

df.isna()

df.head()

df.dtypes

df.corr
