import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score

# Step 1: Data Preprocessing
df = pd.read_csv('transactions.csv')
df = df.dropna() # remove rows with missing values
df['location'] = pd.factorize(df['location'])[0] # encode categorical variable
scaler = StandardScaler()
df[['amount', 'time']] = scaler.fit_transform(df[['amount', 'time']]) # scale numerical variables

# Step 2: Feature Selection/Engineering
X = df[['amount', 'location', 'time']].values

# Step 3: Unsupervised Learning Model Selection
kmeans = KMeans(n_clusters=5, random_state=42)

# Step 4: Train the Unsupervised Model
kmeans.fit(X)

# Step 5: Identify Anomalies
y_pred = kmeans.predict(X)
df['cluster'] = y_pred
fraud_cluster = df.groupby('cluster')['amount'].mean().idxmax()
fraud_indices = df[df['cluster'] == fraud_cluster].index.values

# Step 6: Evaluate Model Performance
y_true = np.zeros(len(df))
y_true[fraud_indices] = 1
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

# Step 7: Iterate and Improve
# This can be done by experimenting with different feature selection/engineering methods, unsupervised learning models, and hyperparameters.
