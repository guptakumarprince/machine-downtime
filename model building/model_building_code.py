###################################################################################################################
###########                             Best model      ##########################################################


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    mean_absolute_error, precision_score, recall_score,
    f1_score, accuracy_score, r2_score, mean_squared_error, roc_auc_score
)
from sklearn.pipeline import Pipeline
import joblib
from sqlalchemy import create_engine
from urllib.parse import quote 
from getpass import getpass
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sweetviz # autoEDA
import matplotlib.pyplot as plt # data visualization
from sqlalchemy import create_engine # connect to SQL database
from feature_engine.outliers import Winsorizer
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer
import scipy.stats as stats
import pylab
from scipy import stats
from sklearn.preprocessing import PowerTransformer


df1 = pd.read_csv(r"C:\Users\Prince Kumar Gupta\OneDrive\Documents\project_148 document\model building\Machine_Downtime.csv")

numeric_features = df1.select_dtypes(exclude = ['object']).columns

numeric_features

catagorical_feature = df1.select_dtypes(include = ['object']).columns

catagorical_feature

prince = pd.DataFrame(df1[numeric_features])

jatin = pd.DataFrame(df1[catagorical_feature])

##### converting data into float #######

prince = prince.astype('float32')
prince.dtypes
print(prince)

## missing value count ####

prince.isna().sum()

jatin.isna().sum()

## duplicate value in row #####

prince.duplicated()
prince.duplicated().sum()

### duplicate column ######

p=prince.corr()

### mean imputatation ########

mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

prince.mean()

prince1 = pd.DataFrame(mean_imputer.fit_transform(prince))

prince1.isna().sum()

new_column_names = ['Hydraulic_Pressure', 'Coolant_Pressure', 'Air_System_Pressure',
                    'Coolant_Temperature', 'Hydraulic_Oil_Temperature',
                    'Spindle_Bearing_Temperature', 'Spindle_Vibration',
                    'Tool_Vibration', 'Spindle_Speed', 'Voltage', 'Torque', 'Cutting']

###### Replace existing column names with new column names ########
prince1.columns = new_column_names

print(prince)

###### outlier treatment ###############
prince1.dtypes

########## Let's find outliers in prince1 ##########

prince1.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()  


# Detection of outliers (find limits for salary based on IQR)
IQR = prince1.quantile(0.75) - prince1.quantile(0.25)

lower_limit = prince1.quantile(0.25) - (IQR * 1.5)
upper_limit = prince1.quantile(0.75) + (IQR * 1.5)

############### 3. Winsorization ###############

# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5, 
                          variables = ['Hydraulic_Pressure', 'Coolant_Pressure', 'Air_System_Pressure', 'Coolant_Temperature', 'Hydraulic_Oil_Temperature', 'Spindle_Bearing_Temperature', 'Spindle_Vibration', 'Tool_Vibration', 'Spindle_Speed', 'Voltage', 'Torque', 'Cutting'])

prince2 = winsor_iqr.fit_transform(prince1[['Hydraulic_Pressure', 'Coolant_Pressure', 'Air_System_Pressure', 'Coolant_Temperature', 'Hydraulic_Oil_Temperature', 'Spindle_Bearing_Temperature', 'Spindle_Vibration', 'Tool_Vibration', 'Spindle_Speed', 'Voltage', 'Torque', 'Cutting']])

## checking outlier #######
prince2.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()  
# **No outliers observed**

## Now Concatenate prince2 and jatin #####

dataset = pd.concat([prince2, jatin], axis=1)
dataset


##moving data to mysql till winsorization

user_name = 'root'
database = '360db'
your_password = 'prince123'
engine = create_engine(f'mysql+pymysql://{user_name}:%s@localhost:3306/{database}' % quote(f'{your_password}'))

dataset.to_sql('machine_downtime', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


### Read the Table (data) from MySQL database
sql = 'SELECT * FROM machine_downtime'

# Convert columns to the best possible dtypes using
df = pd.read_sql_query(sql, engine)

# Assume df and other preprocessing steps are already done

# Perform label encoding for the output variable
label_encoder = LabelEncoder()
df['Downtime'] = label_encoder.fit_transform(df['Downtime'])
joblib.dump(label_encoder, 'label_encoder.pkl')
# Apply transformations
df['Coolant_Temperature'] = df['Coolant_Temperature'].apply(lambda x: np.exp(x))
df['Cutting'] = df['Cutting'].apply(lambda x: np.log(x) if x > 0 else 0)  # Adding a condition to handle log(0)

X = df.drop(columns=['Date', 'Machine_ID', 'Assembly_Line_No', 'Downtime'], axis=1)
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()

# Include StandardScaler and transformations in the pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', scaler),
])

X[numeric_columns] = preprocessing_pipeline.fit_transform(X[numeric_columns])

y = df['Downtime']

# Step 2: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define Models
models = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'NaiveBayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Bagging': BaggingClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'LogisticRegression': LogisticRegression(),
    'MLP': MLPClassifier()
    # Add more models as needed
}

# DataFrame to store results
results = []

# Variables to store information about the best model
best_model_name = None
best_test_accuracy = 0
best_model = None

# Step 4: Hyperparameter Tuning and Evaluation
for name, model in models.items():
    # Include the preprocessing pipeline in the model pipeline
    model_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('model', model),
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model_pipeline.predict(X_train)
    y_test_pred = model_pipeline.predict(X_test)
    
    # Calculate metrics
    train_mape = mean_absolute_error(y_train, y_train_pred)
    test_mape = mean_absolute_error(y_test, y_test_pred)
    
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Additional metrics
    r2 = r2_score(y_test, y_test_pred)
    rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    
    # ROC-AUC is applicable for binary classification, adjust accordingly if needed
    if len(set(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_test_pred)
    else:
        roc_auc = None
    
    results.append({
        'Model': name,
        'MAPE': test_mape,
        'MAE': test_mape,  # MAE and MAPE are the same in this context
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'R2 Score': r2,
        'RMSE': rmse,
        'ROC-AUC': roc_auc
    })
    
    # Save the best model based on test accuracy
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_model_name = name
        best_model = model_pipeline

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv('model_metrics.csv', index=False)

# Save the best model along with the preprocessing steps
joblib.dump(best_model, 'best_model_with_preprocessing.pkl')

