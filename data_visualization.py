# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 23:31:02 2022

@author: zehra
"""

#MATPLOTLIB % SEABORN

#MATPLOTLIB

#Categorical variable: column chart. counting stick
#Numerical selection: history, boxplot


#Categorical Variable Visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df['sex'].value_counts().plot(kind='bar')
plt.show()



#Numeric Variable Visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()


