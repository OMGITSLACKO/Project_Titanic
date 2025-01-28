#!/usr/bin/env python
# coding: utf-8

# # Titanic survivor prediction

# Imports:

# In[1]:


import warnings
import joblib
import requests
from io import StringIO

# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


# Scikit-learn utilities
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# !conda update --all -y
# 

# Link to data:

# In[2]:


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

data = pd.read_csv(url)
data.head()


# ## EDA

# In[3]:


print("Dataset Shape:", data.shape)
print("\nDataset Info:")
data.info()


print("\nMissing Values in Each Column:")
print(data.isnull().sum())


# In[4]:


print("\nSummary Statistics:")
print(data.describe())


print("\nUnique Values per Column:")
for column in data.columns:
    print(f"{column}: {data[column].nunique()} unique values")


# In[5]:


plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


# I think I will drop the cabin, and I'm not sure I want to plug in data to the "age" since that is one of the key variables.

# In[6]:


# Display the number of rows before dropping
print(f"Initial dataset shape: {data.shape}")

# Drop the 'Cabin' column
data_cleaned = data.drop(columns=['Cabin'])
print("Dropped 'Cabin' column.")

# Display the number of missing 'Age' values
missing_age = data_cleaned['Age'].isnull().sum()
print(f"Number of missing 'Age' values before dropping: {missing_age}")

# Drop rows with missing 'Age' values
data_cleaned = data_cleaned.dropna(subset=['Age'])
print("Dropped rows with missing 'Age' values.")

# Display the dataset shape after dropping
print(f"Dataset shape after dropping: {data_cleaned.shape}")


# In[7]:


age_threshold = 21

# Create the 'Child' column: True if Age < 16, else False
data_cleaned['Child'] = data_cleaned['Age'] < age_threshold
print("Added 'Child' column based on 'Age'.")

# Display the distribution of the 'Child' feature
child_counts = data_cleaned['Child'].value_counts()
print("\nDistribution of 'Child' feature:")
print(child_counts)


# I will also drop the unnecessary columns:

# In[8]:


columns_to_drop = ['PassengerId', 'Name', 'Ticket']  # Ticket is like a code, we don't need it, FARE is the one that is interesting

X = data_cleaned.drop(columns=columns_to_drop + ['Survived'])  # 'Survived' is the target
y = data_cleaned['Survived']

print("Updated Features shape:", X.shape)
print("Updated Target shape:", y.shape)


# noice

# In[9]:


data = data_cleaned


# In[10]:


data.head()


# ## Lets check the outliers, and do some plots:

# In[11]:


numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create histograms and density plots
for feature in numerical_features:
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data[feature], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {feature}')
    
    # Density Plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(data[feature], shade=True, color='navy')
    plt.title(f'Density Plot of {feature}')
    
    plt.show()


# In[12]:


for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[feature], color='lightgreen')
    plt.title(f'Boxplot of {feature}')
    plt.show()


# there is one guy with 500 so we will drop that one

# ## Now we encode the one that need encoding to numeral, and also normalize the numerical ones: 

# In[13]:


categorical_features = ['Sex', 'Embarked', 'Pclass']

# Convert 'Pclass' to string to treat it as a categorical variable
X['Pclass'] = X['Pclass'].astype(str)

# Perform one-hot encoding
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
print("Performed one-hot encoding on categorical features.")

# Initialize StandardScaler
scaler = StandardScaler()

# Identify numerical features after encoding
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']

# Scale numerical features
X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])
print("Scaled numerical features.")

# Display the first few rows of the processed data
X_encoded.head()


# In[14]:


# Identify all boolean columns
boolean_columns = X_encoded.select_dtypes(include='bool').columns

# Convert boolean columns to numeric (0/1)
X_encoded[boolean_columns] = X_encoded[boolean_columns].astype(int)

X = X_encoded
X.head()


# In[15]:


# Dropping outliers
outlier_indices = data_cleaned[data_cleaned['Fare'] >= 500].index
data_cleaned = data_cleaned.drop(index=outlier_indices)
X_encoded = X_encoded.drop(index=outlier_indices)
y = y.drop(index=outlier_indices)


# In[16]:


y.head()


# In[17]:


outlier_fare = data_cleaned[data_cleaned['Fare'] >= 500]
print("Outliers with Fare >= 500:")
print(outlier_fare)

outlier_indices = outlier_fare.index
print(f"Indices of outliers to be dropped: {outlier_indices.tolist()}")

# Drop the outlier(s) from data_cleaned
data_cleaned = data_cleaned.drop(index=outlier_indices)

# Update X_encoded and y_final accordingly
X = X_encoded.drop(index=outlier_indices)
y = y.drop(index=outlier_indices)

print(f"Dataset shape after dropping outliers: {X_encoded.shape}")
print(f"Target shape after dropping outliers: {y.shape}")




# In[18]:


# Compute the correlation matrix
corr_matrix = X_encoded.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, square=True, linewidths=.5)
plt.title('Correlation Matrix of Features')
plt.show()


# ## SPLITTING:

# In[19]:


df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
    X, y, test_size=0.2, random_state=52, stratify=y)

print(f"Training set size: {df_X_train.shape}")
print(f"Test set size: {df_X_test.shape}")
print(f"Target set size: {df_y_train.shape}")
print(f"Target set size: {df_y_test.shape}")


# ### Here is what I want to figure out: What was the age where they were considered a child? In other words, when is there a bump up in survivability rate?

# In[20]:


def find_child_age_cutoff_v2(df, age_column='Age', target_column='Survived', age_min=1, age_max=100):
    """
    Determines the age cutoff where the largest increase in survivability occurs by training separate
    logistic regression models for each age threshold.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        age_column (str): The name of the age column.
        target_column (str): The name of the target column indicating survival.
        age_min (int): The minimum age to consider.
        age_max (int): The maximum age to consider.
    
    Returns:
        cutoff_age (int): The age at which the largest increase in survivability occurs.
        results_df (pd.DataFrame): DataFrame containing age cutoffs and corresponding survival probabilities.
    """
    survival_probs = []
    ages = np.arange(age_min, age_max + 1)
    
    for age in ages:
        # Create binary feature: is_child
        df['is_child'] = df[age_column] <= age
        
        # Prepare features and target
        X = df[['is_child']]
        y = df[target_column]
        
        # Initialize and fit logistic regression model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        # Predict survival probability for is_child=True
        prob_child_survival = model.predict_proba([[1]])[0][1]
        survival_probs.append(prob_child_survival)
    
    # Create a DataFrame for results
    results_df = pd.DataFrame({
        'Age_Cutoff': ages,
        'Survival_Probability_Children': survival_probs
    })
    
    # Calculate differences between consecutive ages
    results_df['Survival_Probability_Diff'] = results_df['Survival_Probability_Children'].diff()
    
    # Identify the age with the maximum increase in survivability
    cutoff_age = results_df['Survival_Probability_Diff'].idxmax()  # Index of max diff
    cutoff_age_value = results_df.loc[cutoff_age, 'Age_Cutoff']
    
    # Plot Survival Probability vs Age Cutoff
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x='Age_Cutoff', y='Survival_Probability_Children', marker='o')
    plt.title('Survival Probability of Children by Age Cutoff')
    plt.xlabel('Age Cutoff')
    plt.ylabel('Predicted Survival Probability')
    plt.grid(True)
    plt.axvline(x=cutoff_age_value, color='red', linestyle='--', label=f'Cutoff Age = {cutoff_age_value}')
    plt.legend()
    plt.show()
    
    # Plot Difference in Survival Probability
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Age_Cutoff', y='Survival_Probability_Diff', palette='viridis')
    plt.title('Change in Survival Probability Between Consecutive Age Cutoffs')
    plt.xlabel('Age Cutoff')
    plt.ylabel('Change in Survival Probability')
    plt.grid(True)
    plt.axvline(x=cutoff_age_value, color='red', linestyle='--', label=f'Cutoff Age = {cutoff_age_value}')
    plt.legend()
    plt.show()
    
    print(f"The age cutoff with the largest increase in survivability is at age {cutoff_age_value}.")
    
    # Drop the temporary 'is_child' column
    df.drop(columns=['is_child'], inplace=True)
    
    return cutoff_age_value, results_df


# In[21]:


cutoff_age, results_df = find_child_age_cutoff_v2(
    data, 
    age_column='Age', 
    target_column='Survived', 
    age_min=1, 
    age_max=40
)


# ### Looky looky! IT really starts going up at 21! I find that relevant, thus, I will now go back and change the child age to 21

# ## Lets find the best model:

# In[22]:


import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    PassiveAggressiveClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.impute import SimpleImputer

def find_best_model_classification(df_X_train, df_X_test, df_y_train, df_y_test):
    """
    Trains multiple classification models and selects the best one based on ROC AUC or F1 Score.
    
    Parameters:
    - df_X_train: Training features
    - df_X_test: Testing features
    - df_y_train: Training labels
    - df_y_test: Testing labels
    """
    # Suppress convergence warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    # Handle missing values by imputing with mean for numerical features
    imputer = SimpleImputer(strategy='mean')
    df_X_train = imputer.fit_transform(df_X_train)
    df_X_test = imputer.transform(df_X_test)
    
    # Try importing additional models; if not installed, they will be skipped.
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        LGBMClassifier = None
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        CatBoostClassifier = None

    # Define a list of diverse classification models and configurations:
    models = [
        ("Logistic Regression (L2, C=1)", LogisticRegression(penalty='l2', C=1, solver='lbfgs', 
                                                              max_iter=1000, random_state=42)),
        ("Logistic Regression (L2, C=0.1)", LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', 
                                                                max_iter=1000, random_state=42)),
        ("Logistic Regression (L2, C=10)", LogisticRegression(penalty='l2', C=10, solver='lbfgs', 
                                                               max_iter=1000, random_state=42)),
        ("Logistic Regression (L1, C=1)", LogisticRegression(penalty='l1', C=1, solver='liblinear', 
                                                              max_iter=1000, random_state=42)),
        ("Decision Tree Classifier", DecisionTreeClassifier(random_state=42)),
        ("Random Forest Classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Extra Trees Classifier", ExtraTreesClassifier(random_state=42)),
        ("Gradient Boosting Classifier", GradientBoostingClassifier(random_state=42)),
        ("AdaBoost Classifier", AdaBoostClassifier(random_state=42)),
        ("Bagging Classifier", BaggingClassifier(random_state=42)),
        ("SVC", SVC(probability=True, random_state=42)),
        ("Linear SVC", LinearSVC(max_iter=1000, random_state=42)),
        ("K-Nearest Neighbors Classifier", KNeighborsClassifier()),
        ("Gaussian NB", GaussianNB()),
        ("Bernoulli NB", BernoulliNB()),
        ("Ridge Classifier", RidgeClassifier()),
        ("Passive Aggressive Classifier", PassiveAggressiveClassifier(max_iter=1000, random_state=42)),
        ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
        ("MLP Classifier", MLPClassifier(max_iter=1000, random_state=42))
    ]
    
    # Add additional models if available
    if XGBClassifier is not None:
        models.append(("XGBoost Classifier", 
                       XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)))
    if LGBMClassifier is not None:
        models.append(("LightGBM Classifier", LGBMClassifier(random_state=42)))
    if CatBoostClassifier is not None:
        models.append(("CatBoost Classifier", CatBoostClassifier(verbose=0, random_state=42)))
    
    best_model = None
    best_score = -np.inf
    best_model_instance = None
    best_score_type = None  # To keep track of whether the best_score is ROC AUC or F1

    for name, model in models:
        print(f"Training {name}...")
        try:
            # Train the model
            model.fit(df_X_train, df_y_train)
            
            # Make predictions
            y_pred = model.predict(df_X_test)
            
            # Initialize scores
            roc_auc = None
            f1 = None
            
            # Attempt to calculate ROC AUC if possible
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(df_X_test)[:, 1]
                if not np.isnan(y_pred_proba).any():
                    roc_auc = roc_auc_score(df_y_test, y_pred_proba)
                else:
                    print(f"Warning: {name} predict_proba returned NaN values. ROC AUC will be skipped.")
            elif hasattr(model, "decision_function"):
                # Some models have decision_function instead of predict_proba
                y_scores = model.decision_function(df_X_test)
                if not np.isnan(y_scores).any():
                    roc_auc = roc_auc_score(df_y_test, y_scores)
                else:
                    print(f"Warning: {name} decision_function returned NaN values. ROC AUC will be skipped.")
            
            # Calculate other metrics
            acc = accuracy_score(df_y_test, y_pred)
            f1 = f1_score(df_y_test, y_pred, zero_division=0)
            
            print(f"{name} Model:")
            print(f"Accuracy: {acc:.4f}")
            if roc_auc is not None:
                print(f"ROC AUC Score: {roc_auc:.4f}")
            else:
                print("ROC AUC Score: Not available")
            print(f"F1 Score: {f1:.4f}")
            print("Confusion Matrix:")
            print(confusion_matrix(df_y_test, y_pred))
            print("Classification Report:")
            print(classification_report(df_y_test, y_pred, zero_division=0))
            print("\n")
            
            # Update best model based on ROC AUC if available; otherwise, use F1-score as backup
            if roc_auc is not None:
                if roc_auc > best_score:
                    best_score = roc_auc
                    best_model = name
                    best_model_instance = model
                    best_score_type = 'ROC AUC'
            else:
                if f1 > best_score:
                    best_score = f1
                    best_model = name
                    best_model_instance = model
                    best_score_type = 'F1 Score'
        
        except Exception as e:
            print(f"An error occurred while training {name}: {e}\n")
            continue  # Skip to the next model

    if best_model is not None:
        print(f"The best model is: {best_model} with a {best_score_type} of {best_score:.4f}\n")
        
        # Print the metrics for the best model
        y_pred_best = best_model_instance.predict(df_X_test)
        best_acc = accuracy_score(df_y_test, y_pred_best)
        best_f1 = f1_score(df_y_test, y_pred_best, zero_division=0)
        
        print("Metrics of the Best Model:")
        print(f"Accuracy: {best_acc:.4f}")
        print(f"F1 Score: {best_f1:.4f}")
        
        # Calculate ROC AUC if possible
        if hasattr(best_model_instance, "predict_proba"):
            y_pred_proba_best = best_model_instance.predict_proba(df_X_test)[:, 1]
            if not np.isnan(y_pred_proba_best).any():
                best_roc_auc = roc_auc_score(df_y_test, y_pred_proba_best)
                print(f"ROC AUC Score: {best_roc_auc:.4f}")
            else:
                print("ROC AUC Score: Not available (NaN values in predict_proba)")
        elif hasattr(best_model_instance, "decision_function"):
            y_scores_best = best_model_instance.decision_function(df_X_test)
            if not np.isnan(y_scores_best).any():
                best_roc_auc = roc_auc_score(df_y_test, y_scores_best)
                print(f"ROC AUC Score: {best_roc_auc:.4f}")
            else:
                print("ROC AUC Score: Not available (NaN values in decision_function)")
        else:
            print("ROC AUC Score: Not available (No predict_proba or decision_function)")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(df_y_test, y_pred_best))
        print("Classification Report:")
        print(classification_report(df_y_test, y_pred_best, zero_division=0))
    else:
        print("No suitable model was found.")



# In[23]:


find_best_model_classification(df_X_train, df_X_test, df_y_train, df_y_test)


# ###  First Result
# 
# The best model is: Gradient Boosting Classifier with a ROC AUC Score of 0.8672413793103447
# 
# Metrics of the best model:
# 
# Accuracy: 0.7972027972027972
# 
# F1 Score: 0.7128712871287128
# 
# ROC AUC Score: 0.8672413793103447
# 

# In[24]:


def forward_feature_selection(df, target_column, test_size=0.3, auc_threshold=0.01, max_features=None):
    """
    Perform forward feature selection for a classification task.
    
    This function iteratively evaluates which explanatory variable improves the ROC AUC
    the most when added to the current feature set, using a Logistic Regression classifier.
    
    Parameters:
        df (pd.DataFrame): The full DataFrame containing all features and the target.
        target_column (str): The name of the target column.
        test_size (float): Fraction of the data to use for testing in each iteration.
        auc_threshold (float): Minimum improvement in ROC AUC required to continue adding features.
        max_features (int or None): Maximum number of features to select. If None, all features are considered.
        
    Returns:
        selected_features (list): List of selected feature names.
        metrics_history (dict): History of the selection process with iteration index as key.
    """
   

    # Separate out target and features
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Initialize lists and metrics storage
    remaining_features = list(X.columns)
    selected_features = []
    best_auc_global = 0  # Starting from 0
    metrics_history = {}

    # Set maximum number of features to select if not provided
    if max_features is None:
        max_features = len(remaining_features)

    for i in range(max_features):
        best_auc_this_round = 0
        best_feature = None
        
        # Test each remaining feature to see which gives the best improvement
        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            X_subset = X[candidate_features]
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=test_size, random_state=42)
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            current_auc = roc_auc_score(y_test, y_pred_proba)
            if current_auc > best_auc_this_round:
                best_auc_this_round = current_auc
                best_feature = feature
        
        # Check if the best improvement in this round is significant enough; if not, stop selecting further features.
        if best_auc_this_round - best_auc_global < auc_threshold:
            print(f"Stopping: Improvement in ROC AUC below threshold of {auc_threshold}.")
            break
        
        # Update selected features and metrics
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        best_auc_global = best_auc_this_round
        metrics_history[len(selected_features)] = {"Feature": best_feature, "ROC_AUC": best_auc_global}
        print(f"Iteration {len(selected_features)}: Selected '{best_feature}' with ROC AUC = {best_auc_global:.4f}")
    
    return selected_features, metrics_history


# In[25]:


def combine_X_y(X, y, target_name="Survived"):
    """
    Combines feature matrix X and target vector y into a single DataFrame.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or pd.DataFrame): Target vector.
        target_name (str): Name of the target column in the combined DataFrame.

    Returns:
        pd.DataFrame: Combined DataFrame containing both features and target.
    """
    # Ensure that y is a DataFrame (convert if necessary)
    if isinstance(y, pd.Series):
        y = y.to_frame(name=target_name)
    elif isinstance(y, pd.DataFrame) and target_name not in y.columns:
        y = y.rename(columns={y.columns[0]: target_name})

    # Reset indices to ensure alignment
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Concatenate X and y
    combined_df = pd.concat([X, y], axis=1)

    return combined_df

# Example Usage
df = combine_X_y(X, y)
print("Combined DataFrame shape:", df.shape)
print(df.head())


# In[26]:


selected_features, metrics_history = forward_feature_selection(df, target_column='Survived', 
                                                                test_size=0.3, auc_threshold=0.01, max_features=None)


# welp, okay, let's see

# In[27]:


selected_columns = ['Sex_male', 'Pclass_3', 'Survived']
df_1 = df[selected_columns]

# Split the new DataFrame into training and testing sets
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
    df_1.drop(columns=['Survived']), 
    df_1['Survived'], 
    test_size=0.3, 
    random_state=52, 
    stratify=df_1['Survived']
)

# Display the shapes of the splits
print(f"Training set shape: {df_X_train.shape}, {df_y_train.shape}")
print(f"Testing set shape: {df_X_test.shape}, {df_y_test.shape}")


# In[28]:


find_best_model_classification(df_X_train, df_X_test, df_y_train, df_y_test)


# ### Second Result:
# 
# 
# The best model is: MLP Classifier with a ROC AUC Score of 0.8507449127906976
# 
# Metrics of the best model:
# 
# Accuracy: 0.794392523364486
# 
# F1 Score: 0.6716417910447762
# 
# ROC AUC Score: 0.8507449127906976
# 

# In[29]:


selected_features, metrics_history = forward_feature_selection(df, target_column='Survived', 
                                                                test_size=0.3, auc_threshold=0.001, max_features=None)


# In[30]:


selected_features = ['Sex_male', 'Pclass_3', 'SibSp', 'Fare', 'Child', 'Pclass_2']

# Create a new DataFrame with the selected features and the target variable
df_2 = df[selected_features + ['Survived']]

# Split the new DataFrame into training and testing sets
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
    df_2.drop(columns=['Survived']), 
    df_2['Survived'], 
    test_size=0.3, 
    random_state=52, 
    stratify=df_2['Survived']
)

# Display the shapes of the splits
print(f"Training set shape: {df_X_train.shape}, {df_y_train.shape}")
print(f"Testing set shape: {df_X_test.shape}, {df_y_test.shape}")


# In[31]:


find_best_model_classification(df_X_train, df_X_test, df_y_train, df_y_test)


# ### Third result:
# 
# The best model is: MLP Classifier with a ROC AUC Score of 0.8675508720930233
# 
# Metrics of the best model:
# 
# Accuracy: 0.8177570093457944
# 
# F1 Score: 0.7483870967741936
# 
# ROC AUC Score: 0.8675508720930233

# Looks like we go with number 3

# In[32]:


def test_model(
    df: pd.DataFrame,
    selected_features: list,
    target: str = 'Survived',
    model=None,
    test_size: float = 0.3,
    random_state: int = 52,
    cv: int = 5
) -> dict:
    """
    Trains a classification model, makes predictions, evaluates performance metrics,
    and performs cross-validation.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing features and target.
    - selected_features (list): List of feature column names to be used.
    - target (str): The name of the target variable column. Default is 'Survived'.
    - model: The machine learning model to be trained. If None, RidgeClassifier is used.
    - test_size (float): Proportion of the dataset to include in the test split. Default is 0.3.
    - random_state (int): Random state for reproducibility. Default is 52.
    - cv (int): Number of cross-validation folds. Default is 5.

    Returns:
    - metrics (dict): A dictionary containing all evaluated metrics.
    """

    # Use RidgeClassifier if no model is provided
    if model is None:
        model = RidgeClassifier()
        print("No model provided. Using RidgeClassifier by default.")
    else:
        print(f"Using model: {model.__class__.__name__}")

    # 1. Select Features and Target
    try:
        df_selected = df[selected_features + [target]]
    except KeyError as e:
        raise KeyError(f"One of the selected features or target not found in DataFrame: {e}")

    # 2. Split the Data into Training and Testing Sets
    X = df_selected.drop(columns=[target])
    y = df_selected[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"\nTraining set shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

    # 3. Train the Model
    model.fit(X_train, y_train)
    print("\nModel training completed.")

    # 4. Make Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 5. Evaluate Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Handle ROC AUC depending on model capabilities
    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_test_proba = model.decision_function(X_test)
    else:
        # If neither method is available, use predictions as probabilities
        y_test_proba = y_test_pred

    roc_auc = roc_auc_score(y_test, y_test_proba)

    # 6. Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    mean_cv_accuracy = cv_scores.mean()

    # 7. Print Metrics
    print("\n=== Model Evaluation Metrics ===")
    print(f"Accuracy on Training Data: {train_accuracy:.4f}")
    print(f"Accuracy on Testing Data: {test_accuracy:.4f}")
    print(f"F1 Score on Training Data: {train_f1:.4f}")
    print(f"F1 Score on Testing Data: {test_f1:.4f}")
    print(f"ROC AUC Score on Testing Data: {roc_auc:.4f}")
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Cross-Validation Accuracy: {mean_cv_accuracy:.4f}")

    # 8. Return Metrics
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'roc_auc': roc_auc,
        'cv_scores': cv_scores,
        'mean_cv_accuracy': mean_cv_accuracy
    }

    return metrics



# In[33]:


if __name__ == "__main__":
    # Assuming you have a DataFrame `df` already loaded
    # Define selected features
    selected_features = ['Sex_male', 'Pclass_3', 'SibSp', 'Fare', 'Child', 'Pclass_2']
    
    # Initialize the best model (MLP Classifier)
    best_model = MLPClassifier(random_state=52)
    
    # Call the testing function
    metrics = test_model(
        df=df,
        selected_features=selected_features,
        target='Survived',
        model=best_model,  # Here you can switch to other classifiers
        test_size=0.3,
        random_state=52,
        cv=5
    )
    


# In[34]:


def Over_Fitting_Inquiry(*dfs, target_column, test_size=0.2, random_state=1, names=None, overfitting_threshold=0.1):
    """
    This function evaluates overfitting and performance-related metrics for multiple DataFrames using the Ridge Classifier.
    It computes key metrics for each dataset, generates a detailed summary, and provides a recommendation for the best model,
    considering both performance and generalization capabilities.

    Key Features:
    - **Model Evaluation**:
      Computes metrics including Training/Test Accuracy, F1 Score, Precision, Recall, ROC AUC (if applicable), and Cross-Validation Accuracy.
    - **Overfitting Analysis**:
      Detects overfitting based on the gap between training and test performance (Performance Gap). Models with significant overfitting are penalized.
    - **Aggregated Score**:
      Combines Test Accuracy, Test F1 Score, Mean CV Accuracy, and Test ROC AUC (if available) into a single score. Penalizes overfitted models.
    - **Analytical Summary**:
      Generates a table summarizing all metrics for easy comparison across datasets.
    - **Best Model Recommendation**:
      Identifies the best-performing model based on the Aggregated Score while ensuring it is not overfitted.

    Parameters:
        *dfs: One or more pandas DataFrames, each containing explanatory variables and the target column.
        target_column (str): The name of the target column (binary classification is required).
        test_size (float): Fraction of the data to allocate for the test split (default is 0.2).
        random_state (int): Random seed for reproducibility (default is 1).
        names (list or None): Optional list of custom names for the DataFrames. If None, default names (e.g., df_1, df_2, ...) are assigned.
        overfitting_threshold (float): Threshold for the acceptable gap between training and test performance 
                                       before penalizing the model's Aggregated Score (default is 0.1).

    Returns:
        results (dict): A dictionary where each key corresponds to a DataFrame name, and each value is a dictionary of metrics:
            - **Training Accuracy**: Accuracy on the training set.
            - **Test Accuracy**: Accuracy on the test set.
            - **Training F1 Score**: F1 Score on the training set.
            - **Test F1 Score**: F1 Score on the test set.
            - **Training Precision**: Precision on the training set.
            - **Test Precision**: Precision on the test set.
            - **Training Recall**: Recall on the training set.
            - **Test Recall**: Recall on the test set.
            - **Training ROC AUC**: ROC AUC on the training set (if applicable).
            - **Test ROC AUC**: ROC AUC on the test set (if applicable).
            - **Cross-Validation Scores**: Array of accuracy scores from 5-fold cross-validation on the training set.
            - **Mean CV Accuracy**: Mean accuracy from cross-validation scores.
            - **Performance Gap**: Difference between Training and Test Accuracy.
            - **Aggregated Score**: Overall score combining key metrics, penalized for overfitting.
            - **Analysis**: A brief interpretation of the model's performance, highlighting generalization or overfitting issues.

    Output:
        - **Detailed Metrics for Each Dataset**:
          Displays all metrics and their values for each DataFrame.
        - **Analytical Summary Table**:
          Provides a tabular comparison of key metrics (Test Accuracy, Test F1 Score, Mean CV Accuracy, Test ROC AUC, Aggregated Score, and Analysis).
        - **Best Model Suggestion**:
          Recommends the model with the highest Aggregated Score while ensuring it generalizes well and is not overfitted.

    Notes:
    - Models flagged as overfitting based on the `overfitting_threshold` will be penalized and deprioritized.
    - If all models exhibit overfitting, the function will recommend revisiting the model complexity or dataset quality.
    - Default scoring for cross-validation is Accuracy; this can be adjusted to other metrics if needed.
    """

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 precision_score, recall_score)
    from sklearn.model_selection import train_test_split, cross_val_score
    
    results = {}
    
    # Handle naming: auto-generate names if none or too few are provided.
    if names is None:
        names = [f"df_{i+1}" for i in range(len(dfs))]
    else:
        names = list(names)
        if len(names) < len(dfs):
            for i in range(len(dfs) - len(names)):
                names.append(f"df_{len(names) + i + 1}")
        elif len(names) > len(dfs):
            print("Warning: More names provided than DataFrames. Extra names will be ignored.")
            names = names[:len(dfs)]
    
    # Process each DataFrame
    for idx, df in enumerate(dfs):
        current_name = names[idx]
        
        # Ensure target exists
        if target_column not in df.columns:
            print(f"Error: Target column '{target_column}' not found in '{current_name}'. Skipping this DataFrame.")
            continue
        
        # Extract features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Check if target is binary
        if y.nunique() != 2:
            print(f"Error: Target column '{target_column}' in '{current_name}' is not binary. Skipping this DataFrame.")
            continue
        
        # Split the data
        df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        # Initialize and train the model
        model = RidgeClassifier()
        model.fit(df_X_train, df_y_train)
        
        # Predictions
        y_train_pred = model.predict(df_X_train)
        y_test_pred = model.predict(df_X_test)
        
        # Basic metrics
        train_accuracy = accuracy_score(df_y_train, y_train_pred)
        test_accuracy = accuracy_score(df_y_test, y_test_pred)
        train_f1 = f1_score(df_y_train, y_train_pred)
        test_f1 = f1_score(df_y_test, y_test_pred)
        train_precision = precision_score(df_y_train, y_train_pred, zero_division=0)
        test_precision = precision_score(df_y_test, y_test_pred, zero_division=0)
        train_recall = recall_score(df_y_train, y_train_pred, zero_division=0)
        test_recall = recall_score(df_y_test, y_test_pred, zero_division=0)
        
        # ROC AUC using decision_function if possible
        try:
            y_train_proba = model.decision_function(df_X_train)
            y_test_proba = model.decision_function(df_X_test)
            train_roc_auc = roc_auc_score(df_y_train, y_train_proba)
            test_roc_auc = roc_auc_score(df_y_test, y_test_proba)
        except Exception as e:
            train_roc_auc = None
            test_roc_auc = None
        
        # Cross-validation on training set
        cv_scores = cross_val_score(model, df_X_train, df_y_train, cv=5, scoring='accuracy')
        cv_mean = np.mean(cv_scores)
        
        # Compute Performance Gap
        performance_gap = train_accuracy - test_accuracy
        
        # -------------------------------
        # Compute Aggregated Score with Penalties
        # -------------------------------
        # Base score is an average of key metrics
        metrics_to_average = [test_accuracy, test_f1, cv_mean]
        if test_roc_auc is not None:
            metrics_to_average.append(test_roc_auc)
        
        # Calculate base score
        base_score = sum(metrics_to_average) / len(metrics_to_average)
        
        # Apply penalty for overfitting
        penalty = 0
        if performance_gap > overfitting_threshold:
            penalty = performance_gap * 0.5  # Adjust the multiplier as needed
            analysis_gap = performance_gap
        else:
            analysis_gap = 0
        
        # Final Aggregated Score
        aggregated_score = base_score - penalty
        
        # Overfitting Analysis
        if performance_gap > overfitting_threshold:
            analysis = (f"Overfitting detected (Performance Gap: {round(performance_gap, 4)}). "
                        "Aggregated score penalized.")
        elif (test_accuracy - train_accuracy) > overfitting_threshold:
            analysis = (f"Possible underfitting (Performance Gap: {round(performance_gap, 4)}). "
                        "Consider model complexity or data quality.")
        else:
            analysis = "Good generalization."
        
        # Store everything in results
        results[current_name] = {
            "Training Accuracy": round(train_accuracy, 4),
            "Test Accuracy": round(test_accuracy, 4),
            "Training F1 Score": round(train_f1, 4),
            "Test F1 Score": round(test_f1, 4),
            "Training Precision": round(train_precision, 4),
            "Test Precision": round(test_precision, 4),
            "Training Recall": round(train_recall, 4),
            "Test Recall": round(test_recall, 4),
            "Training ROC AUC": round(train_roc_auc, 4) if train_roc_auc is not None else "Not Available",
            "Test ROC AUC": round(test_roc_auc, 4) if test_roc_auc is not None else "Not Available",
            "Cross-Validation Scores": [round(score, 4) for score in cv_scores],
            "Mean CV Accuracy": round(cv_mean, 4),
            "Performance Gap": round(performance_gap, 4),
            "Aggregated Score": round(aggregated_score, 4),
            "Analysis": analysis
        }
    
    # -------------------------------
    # Print out individual results
    # -------------------------------
    for name, metrics in results.items():
        print(f"Results for {name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        print("\n")
    
    # -------------------------------
    # Build an analytical summary table
    # -------------------------------
    print("=== Analytical Summary ===")
    summary = []
    for name, metrics in results.items():
        summary.append({
            "Name": name,
            "Test Accuracy": metrics["Test Accuracy"],
            "Test F1 Score": metrics["Test F1 Score"],
            "Mean CV Accuracy": metrics["Mean CV Accuracy"],
            "Test ROC AUC": metrics["Test ROC AUC"],
            "Aggregated Score": metrics["Aggregated Score"],
            "Analysis": metrics["Analysis"]
        })
    
    # Create DataFrame for better formatting
    summary_df = pd.DataFrame(summary)
    
    # Display the summary table
    print(summary_df.to_string(index=False))
    
    # -------------------------------
    # Choose a winner based on Aggregated Score
    # -------------------------------
    if len(summary) > 0:
        # Exclude models that are overfitting
        non_overfitting_models = [model for model in summary if "Overfitting detected" not in model["Analysis"]]
        
        if non_overfitting_models:
            # Select the model with the highest Aggregated Score among non-overfitting models
            best_model = max(non_overfitting_models, key=lambda x: x["Aggregated Score"])
            print("\n=== Best Model Suggestion ===")
            print(
                f"The best model is '{best_model['Name']}' "
                f"with an Aggregated Score of {best_model['Aggregated Score']}. \n"
                "This indicates strong overall performance across Test Accuracy, F1 Score, Mean CV Accuracy, "
                "and (if available) ROC AUC, without signs of overfitting."
            )
        else:
            print("\n=== Best Model Suggestion ===")
            print("All models exhibit signs of overfitting based on the defined threshold. Consider revisiting model complexity or data quality.")
    else:
        print("\nNo valid DataFrames were processed. No winner can be selected.")
    
    return results


# In[35]:


Over_Fitting_Inquiry(df, df_1, df_2, target_column='Survived', test_size=0.2, random_state=52, names=['Model1', 'Model2', 'Model3'], overfitting_threshold=0.1)


# Okily dokily thats df_2 still

# In[36]:


def split_data(df, target_column, test_size=0.2, random_state=52):
    """
    Splits the input DataFrame into training and testing sets.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to split.
        target_column (str): The name of the target column.
        test_size (float): The fraction of the dataset to include in the test split (default is 0.2).
        random_state (int): Random seed for reproducibility (default is 42).
    
    Returns:
        X_train (pd.DataFrame): Training set features.
        X_test (pd.DataFrame): Test set features.
        y_train (pd.Series): Training set target.
        y_test (pd.Series): Test set target.
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return df_X_train, df_X_test, df_y_train, df_y_test


# In[37]:


split_data(df, 'Survived')


# ## Let's do the parameter tuning

# In[38]:


import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def tune_mlp_classifier(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 52,
    param_grid: dict = None,
    scoring: str = 'accuracy',
    cv: int = 5,
    n_jobs: int = -1,
    overfitting_threshold: float = 0.1,
    complexity: int = 5
):
    """
    Fine-tunes an MLPClassifier using GridSearchCV on the provided DataFrame and evaluates it with multiple metrics.
    
    The evaluation uses:
      - Test Accuracy
      - Test F1 Score
      - Mean CV Accuracy (via stratified cross-validation)
      - Test ROC AUC (if available; computed via predict_proba)
      
    An aggregated score is computed as the average of these metrics. If the training-test
    performance gap exceeds 'overfitting_threshold', a penalty is applied.

    The 'complexity' parameter (1 to 10) controls how extensive the parameter grid is:
      - Lower = simpler architectures and fewer hyperparameter variations
      - Higher = more complex architectures and broader hyperparameter variations

    Parameters:
        df (pd.DataFrame): The DataFrame containing features and the target column.
        target_column (str): Name of the target column in df (binary classification).
        test_size (float): Fraction for test split (default 0.2).
        random_state (int): Seed for reproducibility (default 52).
        param_grid (dict or None): Custom grid. If None, dynamically generated based on 'complexity'.
        scoring (str): Scoring metric for GridSearchCV (default 'accuracy').
        cv (int): Number of folds for cross-validation (default 5).
        n_jobs (int): Number of jobs to run in parallel (-1 uses all processors).
        overfitting_threshold (float): Gap threshold for penalizing overfitting (default 0.1).
        complexity (int): 1 to 10, controlling the range/granularity of hyperparameters in the grid (default 5).
        
    Returns:
        best_estimator (Pipeline): The best Pipeline (Scaler + MLPClassifier) model found.
        final_aggregated_score (float): The aggregated score (after overfitting penalty).
        best_params (dict): The best hyperparameters as a dictionary.
    """
    
    # 1. Dynamically build a parameter grid if none is provided
    if param_grid is None:
        # Define hidden_layer_sizes based on complexity
        if complexity <= 3:
            hidden_layer_sizes = [(50,), (100,)]
            alpha_values = [0.0001, 0.001, 0.01]
            activation_options = ['relu', 'tanh']
            solver_options = ['adam']
            learning_rate_inits = [0.001, 0.01]
        elif complexity <= 7:
            hidden_layer_sizes = [(100,), (100, 50), (150, 100, 50)]
            alpha_values = np.logspace(-4, -1, num=6).tolist()  # 0.0001 to 0.1
            activation_options = ['relu', 'tanh', 'logistic']
            solver_options = ['adam', 'sgd']
            learning_rate_inits = [0.001, 0.01, 0.1]
        else:  # complexity 8-10
            hidden_layer_sizes = [(100,), (150,), (100, 100), (150, 100, 50)]
            alpha_values = np.logspace(-4, 1, num=10).tolist()  # 0.0001 to 10
            activation_options = ['relu', 'tanh', 'logistic', 'identity']
            solver_options = ['adam', 'sgd', 'lbfgs']
            learning_rate_inits = [0.0001, 0.001, 0.01, 0.1]
        
        param_grid = {
            'mlp__hidden_layer_sizes': hidden_layer_sizes,
            'mlp__alpha': alpha_values,
            'mlp__activation': activation_options,
            'mlp__solver': solver_options,
            'mlp__learning_rate_init': learning_rate_inits
        }
    
    # 2. Prepare data + stratified split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Check if target is binary
    if y.nunique() != 2:
        raise ValueError(f"Target column '{target_column}' is not binary. It has {y.nunique()} unique values.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    print(f"\nTraining set shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing set shape: {X_test.shape}, {y_test.shape}")
    
    # 3. Define Pipeline with StandardScaler and MLPClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(random_state=random_state, max_iter=200))
    ])
    
    # 4. Define StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # 5. Grid Search on MLPClassifier within the pipeline
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        refit=True
    )
    
    print("\nStarting Grid Search...")
    grid_search.fit(X_train, y_train)
    print("Grid Search Completed.")
    
    # Retrieve best estimator and details
    best_estimator = grid_search.best_estimator_
    best_cv_score = grid_search.best_score_
    best_params = grid_search.best_params_
    
    # 6. Evaluate best estimator on test set
    y_test_pred = best_estimator.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    
    # Overfitting analysis
    y_train_pred = best_estimator.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    performance_gap = train_accuracy - test_accuracy
    
    # 7. Compute Test ROC AUC if possible
    try:
        y_test_proba = best_estimator.predict_proba(X_test)[:, 1]
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
    except Exception as e:
        test_roc_auc = None
        print(f"ROC AUC could not be computed: {e}")
    
    # Cross-validation with best_estimator
    cv_scores = cross_val_score(best_estimator, X_train, y_train, cv=skf, scoring=scoring, n_jobs=n_jobs)
    mean_cv_accuracy = np.mean(cv_scores)
    
    # 8. Compute aggregated score
    metrics_list = [test_accuracy, test_f1, mean_cv_accuracy]
    if test_roc_auc is not None:
        metrics_list.append(test_roc_auc)
    base_score = sum(metrics_list) / len(metrics_list)
    
    # Overfitting penalty
    penalty = 0
    if performance_gap > overfitting_threshold:
        penalty = 0.5 * performance_gap
    final_aggregated_score = base_score - penalty
    
    # 9. Create analysis message
    if performance_gap > overfitting_threshold:
        analysis_message = (f"Overfitting detected (Performance Gap: {performance_gap:.4f}). "
                            f"Penalty applied: {penalty:.4f}.")
    elif (test_accuracy - train_accuracy) > overfitting_threshold:
        analysis_message = (f"Possible underfitting (Performance Gap: {performance_gap:.4f}). "
                            "Test performance is higher than training performance.")
    else:
        analysis_message = "Good generalization."
    
    # 10. Printing the results
    print("\n=== Grid Search Results ===")
    print("Best Parameters:", best_params)
    print(f"Best Cross-Validation Score ({scoring}): {best_cv_score:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Mean CV Accuracy: {mean_cv_accuracy:.4f}")
    if test_roc_auc is not None:
        print(f"Test ROC AUC: {test_roc_auc:.4f}")
    else:
        print("Test ROC AUC: Not Available")
    print(f"Performance Gap (Train - Test Accuracy): {performance_gap:.4f}")
    print(f"Aggregated Score (before penalty): {base_score:.4f}")
    print(f"Overfitting Penalty: {penalty:.4f}")
    print(f"Final Aggregated Score: {final_aggregated_score:.4f}")
    print(f"Analysis: {analysis_message}")
    
    # Detailed classification report
    print("\nDetailed Classification Report on Test Set:")
    print(classification_report(y_test, y_test_pred))
    
    # Grid search CV results
    results_df = pd.DataFrame(grid_search.cv_results_)
    if 'display' in globals():
        from IPython.display import display
        display(results_df.sort_values(by='mean_test_score', ascending=False).head(10))
    else:
        print("\nTop 10 CV Results (sorted by mean_test_score):")
        print(results_df.sort_values(by='mean_test_score', ascending=False).head(10))
    
    print("\n=== Winner Summary ===")
    print(f"The best MLPClassifier is obtained with parameters: {best_params}")
    print(f"Final Aggregated Score: {final_aggregated_score:.4f} ("
          "combining Test Accuracy, Test F1, Mean CV Accuracy, and Test ROC AUC if available, "
          f"with an overfitting penalty if Perf Gap > {overfitting_threshold}).")
    
    return best_estimator, final_aggregated_score, best_params


# Okay, this is a long boy, because MLP takes a looot of energy, so maybe don't run this always. For this reason once it is done, I will put it in markdown.

# best_mlp, aggregated_score, best_hyperparams = tune_mlp_classifier(
#     df=df_2,
#     target_column='Survived',
#     test_size=0.3,                # Using 30% of data for testing
#     random_state=52,              # Ensures reproducibility
#     param_grid=None,              # Let the function generate the grid based on complexity
#     scoring='accuracy',           # Primary metric for grid search
#     cv=5,                         # 5-fold cross-validation
#     n_jobs=-1,                    # Utilize all available CPU cores
#     overfitting_threshold=0.1,    # Threshold for penalizing overfitting
#     complexity=5                  # Moderate complexity level
# )
# 

# The best MLPClassifier is obtained with parameters: 
# {'mlp__activation': 'tanh', 'mlp__alpha': 0.1, 'mlp__hidden_layer_sizes': (100,), 
# 
# 'mlp__learning_rate_init': 0.1, 'mlp__solver': 'sgd'}
# 
# Final Aggregated Score: 0.8166 
# 
# (combining Test Accuracy, Test F1, Mean CV Accuracy, and Test ROC AUC if available, with an overfitting penalty if Perf Gap > 0.1).

# In[39]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")  # To ignore any warnings during model training

def test_tuned_mlp(*dfs, target_column, test_size=0.2, random_state=52, 
                  names=None, overfitting_threshold=0.1, tuned_params=None, cv=5, scoring='accuracy'):
    """
    Tests a tuned MLPClassifier on multiple DataFrames and prints relevant metrics.
    
    Parameters:
        *dfs: One or more pandas DataFrames containing features and the target column.
        target_column (str): The name of the target column (binary classification required).
        test_size (float): Fraction of data to be used as test set (default=0.2).
        random_state (int): Seed for reproducibility (default=52).
        names (list or None): Optional list of names for the DataFrames. If None, defaults to df_1, df_2, etc.
        overfitting_threshold (float): Threshold to identify overfitting based on performance gap (default=0.1).
        tuned_params (dict or None): Dictionary of tuned parameters for MLPClassifier. If None, defaults are used.
        cv (int): Number of cross-validation folds (default=5).
        scoring (str): Scoring metric for cross-validation (default='accuracy').
    
    Returns:
        results (dict): Dictionary containing metrics for each DataFrame.
    """
    
    from sklearn.base import clone
    
    results = {}
    
    # Handle naming: auto-generate names if none or too few are provided.
    if names is None:
        names = [f"df_{i+1}" for i in range(len(dfs))]
    else:
        names = list(names)
        if len(names) < len(dfs):
            for i in range(len(dfs) - len(names)):
                names.append(f"df_{len(names) + i + 1}")
        elif len(names) > len(dfs):
            print("Warning: More names provided than DataFrames. Extra names will be ignored.")
            names = names[:len(dfs)]
    
    # Define default tuned parameters if none are provided
    if tuned_params is None:
        tuned_params = {
            'activation': 'relu',
            'alpha': 0.0001,
            'hidden_layer_sizes': (100,)
        }
        print("No tuned parameters provided. Using default parameters for MLPClassifier.")
    else:
        print(f"Using tuned parameters for MLPClassifier: {tuned_params}")
    
    # Iterate over each DataFrame
    for idx, df in enumerate(dfs):
        current_name = names[idx]
        print(f"\n=== Evaluating on {current_name} ===")
        
        # Check if target column exists
        if target_column not in df.columns:
            print(f"Error: Target column '{target_column}' not found in '{current_name}'. Skipping.")
            continue
        
        # Extract features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Check if target is binary
        if y.nunique() != 2:
            print(f"Error: Target column '{target_column}' in '{current_name}' is not binary. Skipping.")
            continue
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        
        # Define the Pipeline with Scaling and MLPClassifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(random_state=random_state, max_iter=500, **tuned_params))
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        print("Model training completed.")
        
        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred, zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        
        # ROC AUC
        try:
            if hasattr(pipeline.named_steps['mlp'], "predict_proba"):
                y_test_proba = pipeline.predict_proba(X_test)[:, 1]
                y_train_proba = pipeline.predict_proba(X_train)[:, 1]
            elif hasattr(pipeline.named_steps['mlp'], "decision_function"):
                y_test_proba = pipeline.decision_function(X_test)
                y_train_proba = pipeline.decision_function(X_train)
            else:
                y_test_proba = y_test_pred
                y_train_proba = y_train_pred
            train_roc_auc = roc_auc_score(y_train, y_train_proba)
            test_roc_auc = roc_auc_score(y_test, y_test_proba)
        except Exception as e:
            train_roc_auc = None
            test_roc_auc = None
            print(f"ROC AUC could not be computed: {e}")
        
        # Cross-Validation on Training Set
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
        mean_cv_accuracy = cv_scores.mean()
        
        # Performance Gap
        performance_gap = train_accuracy - test_accuracy
        
        # Aggregated Score
        metrics_to_average = [test_accuracy, test_f1, mean_cv_accuracy]
        if test_roc_auc is not None:
            metrics_to_average.append(test_roc_auc)
        base_score = np.mean(metrics_to_average)
        
        # Overfitting Penalty
        penalty = 0
        if performance_gap > overfitting_threshold:
            penalty = 0.5 * performance_gap  # You can adjust the multiplier as needed
            analysis = f"Overfitting detected (Performance Gap: {performance_gap:.4f}). Aggregated score penalized."
        elif (test_accuracy - train_accuracy) > overfitting_threshold:
            analysis = f"Possible underfitting (Performance Gap: {performance_gap:.4f}). Consider model complexity or data quality."
        else:
            analysis = "Good generalization."
        
        # Final Aggregated Score
        aggregated_score = base_score - penalty
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Classification Report
        class_report = classification_report(y_test, y_test_pred)
        
        # Store results
        results[current_name] = {
            "Training Accuracy": round(train_accuracy, 4),
            "Test Accuracy": round(test_accuracy, 4),
            "Training F1 Score": round(train_f1, 4),
            "Test F1 Score": round(test_f1, 4),
            "Training Precision": round(train_precision, 4),
            "Test Precision": round(test_precision, 4),
            "Training Recall": round(train_recall, 4),
            "Test Recall": round(test_recall, 4),
            "Training ROC AUC": round(train_roc_auc, 4) if train_roc_auc is not None else "Not Available",
            "Test ROC AUC": round(test_roc_auc, 4) if test_roc_auc is not None else "Not Available",
            "Cross-Validation Scores": [round(score, 4) for score in cv_scores],
            "Mean CV Accuracy": round(mean_cv_accuracy, 4),
            "Performance Gap": round(performance_gap, 4),
            "Aggregated Score": round(base_score, 4),
            "Overfitting Penalty": round(penalty, 4),
            "Final Aggregated Score": round(aggregated_score, 4),
            "Analysis": analysis,
            "Confusion Matrix": cm,
            "Classification Report": class_report
        }
        
        # Print metrics
        print("\n--- Evaluation Metrics ---")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Training F1 Score: {train_f1:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Training Precision: {train_precision:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Training Recall: {train_recall:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Training ROC AUC: {train_roc_auc:.4f}" if train_roc_auc is not None else "Training ROC AUC: Not Available")
        print(f"Test ROC AUC: {test_roc_auc:.4f}" if test_roc_auc is not None else "Test ROC AUC: Not Available")
        print(f"Mean CV Accuracy: {mean_cv_accuracy:.4f}")
        print(f"Performance Gap (Train - Test Accuracy): {performance_gap:.4f}")
        print(f"Aggregated Score (before penalty): {base_score:.4f}")
        print(f"Overfitting Penalty: {penalty:.4f}")
        print(f"Final Aggregated Score: {aggregated_score:.4f}")
        print(f"Analysis: {analysis}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(class_report)
    
    # Return the results dictionary
    return results


# In[40]:


tuned_params = {
    'activation': 'tanh',
    'alpha': 0.1,
    'hidden_layer_sizes': (100,)
}


results = test_tuned_mlp(
    df_2,
    target_column='Survived',
    test_size=0.2,
    random_state=52,
    names=['Model1', 'Model2', 'Model3'],
    overfitting_threshold=0.1,
    tuned_params=tuned_params
)


# In[41]:


tuned_params = {
    'activation': 'tanh',
    'alpha': 0.1,
    'hidden_layer_sizes': (100,)
}


results = test_tuned_mlp(
    df_1,
    target_column='Survived',
    test_size=0.2,
    random_state=52,
    names=['Model1', 'Model2', 'Model3'],
    overfitting_threshold=0.1,
    tuned_params=tuned_params
)


# In[42]:


tuned_params = {
    'activation': 'tanh',
    'alpha': 0.1,
    'hidden_layer_sizes': (100,)
}


results = test_tuned_mlp(
    df,
    target_column='Survived',
    test_size=0.2,
    random_state=52,
    names=['Model1', 'Model2', 'Model3'],
    overfitting_threshold=0.1,
    tuned_params=tuned_params
)


# False positive and false negatives holds less meaning here, than in - for example - a healthcare modell. So, we try to maximize the accuracy and minimize the false predictions. Thats that. 

# ### Ok let's save it
# 
# Also please note that we save the standard scaler as well, so that it can scale the input when we publish the model. 

# In[43]:


# ============================
# 1. Imports
# ============================

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================
# 2. Load and Prepare the Dataset
# ============================

# Assuming df_2 is already loaded as a DataFrame with preprocessed features
# For example:
# df_2 = pd.read_csv('path_to_preprocessed_data.csv')

df = df_2.copy()

# ============================
# 3. Feature Engineering
# ============================

# Standardize all column names to lowercase to avoid case sensitivity issues
df.columns = [col.lower() for col in df.columns]

print("=== Columns in DataFrame ===")
print(df.columns.tolist())

# ============================
# 4. Define Features and Target
# ============================

feature_columns = ['sex_male', 'pclass_3', 'sibsp', 'fare', 'child', 'pclass_2']

# Check for missing features
missing_features = [col for col in feature_columns if col not in df.columns]
if missing_features:
    print(f"Error: Missing features in DataFrame: {missing_features}")
    exit()

X = df[feature_columns]
y = df['survived']

# ============================
# 5. Split the Dataset
# ============================

test_size = 0.02  # Adjusted to ensure sufficient samples
random_state = 52

print("\n=== Class Distribution Before Split ===")
print(y.value_counts())

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print("\n=== Training Set ===")
    print(X_train.head())
    print(y_train.head())
    print("\n=== Test Set ===")
    print(X_test.head())
    print(y_test.head())
except ValueError as ve:
    print(f"Error during train_test_split: {ve}")
    print("Possible causes:")
    print("- Not enough samples to stratify")
    print("- Some classes have too few samples")
    exit()

# ============================
# 6. Define the Pipeline
# ============================

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        random_state=random_state,
        max_iter=500,
        activation='tanh',
        alpha=0.1,
        hidden_layer_sizes=(100,)
    ))
])

# ============================
# 7. Train the Model
# ============================

pipeline.fit(X_train, y_train)

# ============================
# 8. Evaluate the Model
# ============================

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
train_precision = precision_score(y_train, y_train_pred, zero_division=0)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
train_recall = recall_score(y_train, y_train_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)

try:
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
except AttributeError:
    train_roc_auc = None
    test_roc_auc = None

# Print evaluation metrics
print("\n=== Model Evaluation Metrics ===")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Training F1 Score: {train_f1:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Training Precision: {train_precision:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Training Recall: {train_recall:.4f}")
print(f"Test Recall: {test_recall:.4f}")
if train_roc_auc is not None and test_roc_auc is not None:
    print(f"Training ROC AUC: {train_roc_auc:.4f}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")
else:
    print("ROC AUC Score: Not Available for this model.")

# ============================
# 9. Save the Model
# ============================

model_path = 'best_tuned_mlp_model.joblib'  # Changed extension to .joblib
joblib.dump(pipeline, model_path)
print(f"\nModel saved successfully as '{model_path}'")

# ============================
# 10. Verify the Scaler
# ============================

print("\n=== Verifying the Scaler ===")
print(pipeline.named_steps['scaler'])


# In[44]:


import joblib
import pandas as pd

# ============================
# 1. Load the Trained Pipeline
# ============================

# Define the path to the saved model
model_path = 'best_tuned_mlp_model.joblib'

try:
    # Load the pipeline (which includes the scaler and MLPClassifier)
    pipeline = joblib.load(model_path)
    print(f"Successfully loaded the pipeline from '{model_path}'.\n")
except FileNotFoundError:
    print(f"Error: The model file '{model_path}' was not found.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    exit()

# ============================
# 2. Verify Pipeline Components
# ============================

print("=== Pipeline Components ===")
for step_name, step in pipeline.named_steps.items():
    print(f"{step_name}: {step}")
print("\n")

# ============================
# 3. Define Test Subjects
# ============================

# Test Subject 1: Expected to NOT survive (Drown)
# - Male, 3rd Class, 1 Sibling/Spouse Aboard, Low Fare, Adult
test_subject_1 = {
    'sex_male': 1,  # Male
    'pclass_3': 1,  # 3rd Class
    'sibsp': 1,  # 1 Sibling/Spouse Aboard
    'fare': 10.0,  # Low Fare
    'child': 0,  # Adult
    'pclass_2': 0  # Not 2nd Class
}

# Test Subject 2: Expected to Survive
# - Female, 1st Class, No Siblings/Spouses Aboard, High Fare, Child
test_subject_2 = {
    'sex_male': 0,  # Female
    'pclass_3': 0,  # Not 3rd Class
    'sibsp': 0,  # No Siblings/Spouses Aboard
    'fare': 100.0,  # High Fare
    'child': 1,  # Child
    'pclass_2': 0  # Not 2nd Class
}

# Create a DataFrame with the test subjects
test_data = pd.DataFrame([test_subject_1, test_subject_2])

print("=== Test Subjects ===")
print(test_data)
print("\n")

# ============================
# 4. Apply Scaling and Predict
# ============================

# Access the scaler
scaler = pipeline.named_steps['scaler']

# Manually scale the test data using the scaler
scaled_test_data = scaler.transform(test_data)

# Make predictions using the pipeline
predictions = pipeline.named_steps['mlp'].predict(scaled_test_data)

# Predict probabilities
probabilities = pipeline.named_steps['mlp'].predict_proba(scaled_test_data)

# Map predictions to labels
prediction_labels = ['Did Not Survive', 'Survived']

# ============================
# 5. Display the Results
# ============================

for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
    print(f"--- Test Subject {i+1} ---")
    print(test_data.iloc[i])
    print(f"Prediction: {prediction_labels[pred]}")
    print(f"Probability:")
    print(f"  Did Not Survive: {proba[0]:.4f}")
    print(f"  Survived: {proba[1]:.4f}\n")


# In[45]:


scaled_data = pipeline.named_steps['scaler'].transform(test_data)
predictions = pipeline.named_steps['mlp'].predict(scaled_data)
print(scaled_data)
print(predictions)


# ## LETS UPLOAD IT BABY!!!

# pip freeze
# 

# # The go to CMD:
# 
# pip install pipreqs
# 
# pipreqs C:\Users\Administrator\Project_Titanic
# 

# Dang baby, this pipreqs work :O 

# In[ ]:





# # Upgrade pip
# !pip install --upgrade pip
# 
# # Update the required libraries to the latest versions
# !pip install --upgrade numpy pandas scikit-learn xgboost lightgbm catboost matplotlib seaborn joblib flask
# 

# ! pip install --upgrade pip
# 

# In[ ]:





# In[ ]:





# cd C:\Users\Administrator\Project_Titanic
# 
# 
# 
# 
# conda env list
# 
# 
# conda activate project_titanic
# 
# 
# 
# 
# app.py
# 

# This is how you double check:

# In[51]:


import joblib
import numpy as np

# Load the saved pipeline
model_path = 'best_tuned_mlp_model.joblib'
pipeline_loaded = joblib.load(model_path)
print(f"Model loaded successfully from '{model_path}'.")

# Access and print the scaler
scaler_loaded = pipeline_loaded.named_steps['scaler']
print("\n=== Loaded Scaler ===")
print(scaler_loaded)

# Access and print the classifier
classifier_loaded = pipeline_loaded.named_steps['mlp']
print("\n=== Loaded Classifier ===")
print(classifier_loaded)

# Example prediction
# Replace the following feature values with a realistic example
# Features order: [sex_male, pclass_3, sibsp, fare, child, pclass_2]
example_input = np.array([[1, 1, 2, 32.0, 1, 1]])  # Example feature array
prediction = pipeline_loaded.predict(example_input)
print(f"\nPrediction for the example input: {prediction[0]}")


# test:

# In[52]:


import joblib

# Load the pipeline
pipeline_loaded = joblib.load('best_tuned_mlp_model.joblib')

# Access the scaler
print(pipeline_loaded.named_steps['scaler'])


# another test, for scaler presence:

# In[53]:


import joblib

# Load the pipeline
pipeline_loaded = joblib.load('best_tuned_mlp_model.joblib')

# Access the scaler
print(pipeline_loaded.named_steps['scaler'])


# In[55]:


import joblib
import numpy as np
import pandas as pd

# Load the pipeline
model_path = 'best_tuned_mlp_model.joblib'
pipeline_loaded = joblib.load(model_path)
print(f"Model loaded successfully from '{model_path}'.")

# Access and print the scaler
scaler_loaded = pipeline_loaded.named_steps['scaler']
print("\n=== Loaded Scaler ===")
print(scaler_loaded)

# Access and print the classifier
classifier_loaded = pipeline_loaded.named_steps['mlp']
print("\n=== Loaded Classifier ===")
print(classifier_loaded)

# Example prediction
# Features order: [sex_male, pclass_3, sibsp, fare, child, pclass_2]
example_features = {
    'sex_male': 0,    # Female
    'pclass_3': 0,     # 1st class
    'sibsp': 2,
    'fare': 32.0,
    'child': 0,
    'pclass_2': 0
}
example_df = pd.DataFrame([example_features])
print("\n=== Example Input ===")
print(example_df)

# Make prediction
prediction = pipeline_loaded.predict(example_df)[0]  # 0 or 1
print(f"\nPrediction for the example input: {prediction}")

# Check probabilities
try:
    proba = pipeline_loaded.predict_proba(example_df)[0]
    print(f"Prediction probabilities: {proba}")
except AttributeError:
    print("Prediction probabilities not available for this model.")


# # OKAY!!!! LOOKS LIKE THIS MLP CLASSIFIER WON'T WORK PROPERLY, SO I WILL TRY THIS AGAIN, STARTING FROM DF!!!!

# In[56]:


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

data = pd.read_csv(url)
data.head()


# In[57]:


# Display the number of rows before dropping
print(f"Initial dataset shape: {data.shape}")

# Drop the 'Cabin' column
data_cleaned = data.drop(columns=['Cabin'])
print("Dropped 'Cabin' column.")

# Display the number of missing 'Age' values
missing_age = data_cleaned['Age'].isnull().sum()
print(f"Number of missing 'Age' values before dropping: {missing_age}")

# Drop rows with missing 'Age' values
data_cleaned = data_cleaned.dropna(subset=['Age'])
print("Dropped rows with missing 'Age' values.")

# Display the dataset shape after dropping
print(f"Dataset shape after dropping: {data_cleaned.shape}")


# In[58]:


age_threshold = 21

# Create the 'Child' column: True if Age < 16, else False
data_cleaned['Child'] = data_cleaned['Age'] < age_threshold
print("Added 'Child' column based on 'Age'.")

# Display the distribution of the 'Child' feature
child_counts = data_cleaned['Child'].value_counts()
print("\nDistribution of 'Child' feature:")
print(child_counts)


# In[59]:


columns_to_drop = ['PassengerId', 'Name', 'Ticket']  # Ticket is like a code, we don't need it, FARE is the one that is interesting

X = data_cleaned.drop(columns=columns_to_drop + ['Survived'])  # 'Survived' is the target
y = data_cleaned['Survived']

print("Updated Features shape:", X.shape)
print("Updated Target shape:", y.shape)


# In[60]:


df = pd.concat([X, y], axis=1)
df.columns = list(X.columns) + ['Survived']


# In[61]:


df.head()


# In[62]:


outlier_fare = df[df['Fare'] > 500]
print("Outlier with Fare > 500:")
print(outlier_fare)

# Get the index of the outlier
outlier_index = outlier_fare.index
print(f"Index of the outlier to be dropped: {outlier_index.tolist()}")

# Drop the outlier from the combined DataFrame
df = df.drop(index=outlier_index)

# Confirm the shape of the DataFrame after dropping the outlier
print(f"Shape of the combined DataFrame after dropping outliers: {df.shape}")

# Optional: Plot boxplots for numerical features after cleaning
numerical_features = [col for col in X.columns if col not in ['Child', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Pclass_2', 'Pclass_3']]
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[feature], color='lightgreen')
    plt.title(f'Boxplot of {feature} (after outlier removal)')
    plt.show()


# In[63]:


def preprocess_dataframe(df):
    """
    Enhanced preprocessing that:
    1. Automatically detects feature types
    2. Ensures proper numeric formats
    3. Preserves original column names
    """
    df = df.copy()
    
    # Auto-detect features (modified to handle booleans)
    categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Convert low-cardinality numeric to categorical
    for col in numeric_cols:
        if df[col].nunique() < 10:
            categorical_features.append(col)
            df[col] = df[col].astype(str)

    # Handle missing values
    df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())

    # One-hot encode and ensure numeric values
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Convert all boolean columns to integers (critical fix!)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Scale numerical features
    if numerical_features:
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df


# In[64]:


df = preprocess_dataframe(df)


# In[65]:


df.head()


# In[66]:


from sklearn.model_selection import train_test_split

def split_dataframe(df, target_column, test_size=0.2, random_state=None):
    """
    Split a DataFrame into training and test sets with stratification.
    
    Parameters:
    - df: pandas DataFrame containing both features and target
    - target_column: name of the target column (string)
    - test_size: proportion of dataset to allocate to test (default 0.2)
    - random_state: random seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test as DataFrames/Series
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Ensures same class distribution in splits
    )
    
    return X_train, X_test, y_train, y_test


# In[67]:


df_X_train, df_X_test, df_y_train, df_y_test = split_dataframe(df, "Survived_1", test_size=0.2, random_state=None)

print(f"Training set size: {df_X_train.shape}")
print(f"Test set size: {df_X_test.shape}")
print(f"Target set size: {df_y_train.shape}")
print(f"Target set size: {df_y_test.shape}")


# In[68]:


find_best_model_classification(df_X_train, df_X_test, df_y_train, df_y_test)


# In[69]:


selected_features, metrics_history = forward_feature_selection(df, target_column='Survived_1', 
                                                                test_size=0.2, auc_threshold=0.001, max_features=None)


# In[70]:


selected_features = [
    "Sex_male",
    "Fare",
    "Pclass_3",
    "Child_True",
    "Pclass_2",
    "Age",
    "Embarked_S",
    "SibSp_5",
    "Survived_1"
]

df_1 = df[selected_features]

df_1.head()


# In[71]:


df_X_train, df_X_test, df_y_train, df_y_test = split_dataframe(df_1, "Survived_1", test_size=0.2, random_state=None)

print(f"Training set size: {df_X_train.shape}")
print(f"Test set size: {df_X_test.shape}")
print(f"Target set size: {df_y_train.shape}")
print(f"Target set size: {df_y_test.shape}")


# In[72]:


find_best_model_classification(df_X_train, df_X_test, df_y_train, df_y_test)


# I'm not doing MLP again.

# In[73]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Train Logistic Regression on df_1
log_reg_model = LogisticRegression(penalty='l2', C=10, solver='lbfgs', max_iter=1000, random_state=42)

# Fit the model on the training data
log_reg_model.fit(df_X_train, df_y_train)

# Predict on the test set
y_pred = log_reg_model.predict(df_X_test)
y_pred_proba = log_reg_model.predict_proba(df_X_test)[:, 1]

# Calculate performance metrics
accuracy = accuracy_score(df_y_test, y_pred)
roc_auc = roc_auc_score(df_y_test, y_pred_proba)
conf_matrix = confusion_matrix(df_y_test, y_pred)
class_report = classification_report(df_y_test, y_pred)

# Print results
print("Logistic Regression (L2, C=10) Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


# In[74]:


Over_Fitting_Inquiry(df_1, target_column='Survived_1', test_size=0.2, random_state=52, names=['Model1'], overfitting_threshold=0.1)


# In[75]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

def tune_logistic_regression(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 52,
    param_grid: dict = None,
    scoring: str = 'accuracy',
    cv: int = 5,
    n_jobs: int = -1,
    overfitting_threshold: float = 0.1,
    complexity: int = 5
):
    """
    Fine-tunes a Logistic Regression model with L2 regularization using GridSearchCV.
    
    Features:
    - Dynamic parameter grid based on complexity level (1-10)
    - Built-in feature scaling
    - Overfitting detection and penalty system
    - Comprehensive performance metrics
    - Cross-validation validation

    Parameters:
        df (pd.DataFrame): Input DataFrame with features and target
        target_column (str): Name of the target column (binary classification)
        test_size (float): Test set size (default: 0.2)
        random_state (int): Random seed (default: 52)
        param_grid (dict): Custom parameter grid (optional)
        scoring (str): Optimization metric (default: 'accuracy')
        cv (int): Cross-validation folds (default: 5)
        n_jobs (int): Parallel jobs (default: -1)
        overfitting_threshold (float): Allowed train-test gap (default: 0.1)
        complexity (int): Hyperparameter grid complexity (1-10, default: 5)

    Returns:
        best_estimator: Optimized Logistic Regression model
        final_score: Comprehensive performance score
        best_params: Best hyperparameters
    """
    
    # Validate target
    if df[target_column].nunique() != 2:
        raise ValueError("Target must be binary for logistic regression")

    # Create dynamic parameter grid
    if param_grid is None:
        base_c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        
        # Adjust grid based on complexity
        if complexity <= 3:
            param_grid = {
                'logreg__C': [1, 10, 100],
                'logreg__solver': ['lbfgs', 'liblinear'],
                'logreg__max_iter': [100, 200]
            }
        elif complexity <= 7:
            param_grid = {
                'logreg__C': np.logspace(-2, 3, 20).tolist(),
                'logreg__solver': ['lbfgs', 'liblinear', 'saga'],
                'logreg__max_iter': [200, 500],
                'logreg__class_weight': [None, 'balanced']
            }
        else:
            param_grid = {
                'logreg__C': np.logspace(-4, 4, 50).tolist(),
                'logreg__solver': ['lbfgs', 'newton-cg', 'liblinear', 'saga'],
                'logreg__max_iter': [500, 1000],
                'logreg__class_weight': [None, 'balanced'],
                'logreg__fit_intercept': [True, False],
                'logreg__tol': [1e-4, 1e-5, 1e-6]
            }

    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(penalty='l2', random_state=random_state))
    ])

    # Configure grid search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )

    # Execute grid search
    print("\nStarting Logistic Regression optimization...")
    grid_search.fit(X_train, y_train)
    print("Optimization complete!")

    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate performance
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, train_pred),
        'test_accuracy': accuracy_score(y_test, test_pred),
        'f1': f1_score(y_test, test_pred),
        'roc_auc': roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]),
        'precision': precision_score(y_test, test_pred),
        'recall': recall_score(y_test, test_pred)
    }

    # Calculate overfitting penalty
    performance_gap = metrics['train_accuracy'] - metrics['test_accuracy']
    penalty = 0.5 * max(0, performance_gap - overfitting_threshold)
    
    # Composite score calculation
    base_score = np.mean([
        metrics['test_accuracy'],
        metrics['f1'],
        metrics['roc_auc'],
        grid_search.best_score_
    ])
    final_score = base_score - penalty

    # Print results
    print("\n=== Best Model Performance ===")
    print(f"Best Parameters: {best_params}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")
    print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Overfitting Penalty: {penalty:.4f}")
    print(f"Final Score: {final_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    return best_model, final_score, best_params


# ### Hypertuning the second model:

# In[80]:


best_model, score, params = tune_logistic_regression(
    df=df_1,
    target_column='Survived_1',
    complexity=8  # More complex parameter search
)


# In[77]:


df_1.head()


# In[78]:


import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, classification_report)

# ============================
# 1. Load & Verify Data
# ============================
# Model naming and versioning
model_name = "survival_predictor_v1"
save_formats = ['joblib', 'pkl']

required_features = [
    'Sex_male', 'Fare', 'Pclass_3', 'Child_True',
    'Pclass_2', 'Age', 'Embarked_S', 'SibSp_5'
]

target_column = 'Survived_1'

# Verify dataframe structure
assert all(col in df_1.columns for col in required_features + [target_column]), \
    "Missing required columns in dataframe"

# ============================
# 2. Configure Optimal Model
# ============================
best_params = {
    'C': 0.0062505519252739694,
    'class_weight': None,
    'fit_intercept': True,
    'max_iter': 500,
    'solver': 'lbfgs',
    'tol': 0.0001
}

# Create optimized pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(**best_params, random_state=52))
])

# ============================
# 3. Data Preparation
# ============================
X = df_1[required_features]
y = df_1[target_column]

# Stratified split (maintains class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=52,
    stratify=y
)

# ============================
# 4. Train & Evaluate
# ============================
pipeline.fit(X_train, y_train)

# Generate predictions and probabilities
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_proba),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred)
}

# ============================
# 5. Results Output
# ============================
print("\n" + "="*55)
print(f"=== Optimal Model: {model_name} ===")
print("="*55)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Test F1: {metrics['f1']:.4f}")
print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================
# 6. Enhanced Model Persistence
# ============================
# Save in multiple formats
for fmt in save_formats:
    filename = f"{model_name}.{fmt}"
    joblib.dump(pipeline, filename)
    print(f"Model saved as: {filename}")

# Additional metadata saving
model_info = {
    'model_name': model_name,
    'features': required_features,
    'target': target_column,
    'metrics': metrics,
    'parameters': best_params
}

joblib.dump(model_info, f"{model_name}_metadata.joblib")

print("\n" + "-"*55)
print(f"Model successfully saved in {len(save_formats)} formats")
print(f"Core model name: {model_name}")
print(f"Additional metadata saved: {model_name}_metadata.joblib")
print("Final features used:", required_features)


# In[79]:


import joblib
import pandas as pd

# Load the saved model and metadata
model_name = "survival_predictor_v1"
model = joblib.load(f'{model_name}.joblib')
metadata = joblib.load(f'{model_name}_metadata.joblib')

print(f"\n=== Testing Model: {metadata['model_name']} ===")
print("Features used:", metadata['features'])

# Create test cases with ALL required features in exact order
test_cases = [
    {  # Likely survivor (Female, 1st class, high fare)
        'Sex_male': 0,
        'Fare': 120.0,
        'Pclass_3': 0,
        'Child_True': 0,
        'Pclass_2': 0,
        'Age': 28.0,
        'Embarked_S': 1,
        'SibSp_5': 0
    },
    {  # Likely non-survivor (Male, 3rd class, low fare)
        'Sex_male': 1,
        'Fare': 7.25,
        'Pclass_3': 1,
        'Child_True': 0,
        'Pclass_2': 0,
        'Age': 35.0,
        'Embarked_S': 1,
        'SibSp_5': 0
    }
]

# Convert to DataFrame with proper feature ordering
test_df = pd.DataFrame(test_cases)[metadata['features']]

# Get predictions and probabilities
predictions = model.predict(test_df)
probabilities = model.predict_proba(test_df)

# Display detailed results
print("\n=== Test Case Results ===")
for i, (features, pred, prob) in enumerate(zip(test_df.to_dict('records'), predictions, probabilities)):
    print(f"\nCase {i+1}:")
    print(f"Prediction: {'Survived (1)' if pred == 1 else 'Did Not Survive (0)'}")
    print(f"Probability: {prob[1]:.2%} chance of survival")
    print("Feature Values:")
    print(f" - Sex: {'Male' if features['Sex_male'] == 1 else 'Female'}")
    print(f" - Class: {'3rd' if features['Pclass_3'] else '2nd' if features['Pclass_2'] else '1st'}")
    print(f" - Fare: £{features['Fare']:.2f}")
    print(f" - Age: {features['Age']:.1f} years")
    print(f" - Embarked: {'Southampton' if features['Embarked_S'] else 'Other'}")
    print(f" - SibSp_5: {'Yes' if features['SibSp_5'] else 'No'}")
    print(f" - Child: {'Yes' if features['Child_True'] else 'No'}")
    print("➖" * 40)

print("\nExpected Pattern:")
print("Case 1 should survive (high fare, female, 1st class)")
print("Case 2 should not survive (male, 3rd class, low fare)")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




