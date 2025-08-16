import numpy as np
import pandas as pd

data = {
    'Social_media_followers':[1000000,np.nan,2000000,1310000,1700000,np.nan,4100000.0,1600000.0,2200000.0,1000000.0],
    'Sold_out':[1,0,0,1,0,0,0,1,0,1]

}
data_frame = pd.DataFrame(data=data)
print(data_frame)

X= data_frame[['Social_media_followers']]
y = data_frame[['Sold_out']]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=19)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.pipeline import make_pipeline
pipe1 = make_pipeline(imputer,lr)


print(pipe1.fit(X_train,y_train))

print(pipe1.score(X_train,y_train))

print(pipe1.score(X_test,y_test))

print(pipe1.named_steps.simpleimputer.statistics_)

print(pipe1.named_steps.logisticregression.coef_)

# More Advanced Pipeline
print("More Advanced Pipeline")
data2 = {
    'Genre':['Rock','Metal','Bluegrass','Rock',np.nan,'Rock','Rock',np.nan,'Bluegrass','Rock'],
    'Social_media_followers':[1000000,np.nan,2000000,1310000,1700000,np.nan,4100000.0,1600000.0,2200000.0,1000000.0],
    'Sold_out':[1,0,0,1,0,0,0,1,0,1]
}

data_frame_2 = pd.DataFrame(data=data2)
X = data_frame_2.iloc[:,0:2]
y = data_frame_2.iloc[:,2]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=19)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

nums_col = ['Social_media_followers']
cat_cols = ['Genre']

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline(steps=[('impute',SimpleImputer(strategy='mean')),
                              ('scale',StandardScaler())])
from sklearn.preprocessing import OneHotEncoder
cat_pipeline = Pipeline(steps= [
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('one-hot-encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))

])

from sklearn.compose import ColumnTransformer
col_transformer = ColumnTransformer(
    transformers=[
        ('num_pipeline',num_pipeline,nums_col),
        ('cat_pipeline',num_pipeline,nums_col)
    ],
    remainder='drop',
    n_jobs=-1
)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
pipefinal = make_pipeline(col_transformer,dtc)
print(pipefinal.fit(X_train,y_train))

print(pipefinal.score(X_test,y_test))

# How to save your pipeline
import joblib
print(joblib.dump(pipefinal,"pipe.joblib"))

pipefinal2 = joblib.load("pipe.joblib")


