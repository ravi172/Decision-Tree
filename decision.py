import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

df = pd.read_excel('RIDGE.xlsx')

df.head()

##converting data type

df['base_price'] = pd.to_numeric(df['base_price'])
df['original_price'] = pd.to_numeric(df['original_price'])
df['discount_amount'] = pd.to_numeric(df['discount_amount'])
df['price'] = pd.to_numeric(df['price'])
#df['product_type'] = pd.to_numeric(df['product_type'])
##df = df.drop(['product_type'], axis=1)

df.dtypes
##df.head(6)
##df['base_price'].dtype

from sklearn.model_selection import train_test_split

X = df.drop('qty_ordered',axis=1)
y = df['qty_ordered']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.tree import DecisionTreeClassifier

X_train.dtypes

dtree = DecisionTreeClassifier(criterion = 'entropy')

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

conf_matrix=confusion_matrix(y_test,predictions)
accuracy=accuracy_score(y_test,predictions)

conf_matrix,accuracy

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
