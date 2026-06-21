import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

# Create models directory
os.makedirs('models', exist_ok=True)

# Load data
dataset = pd.read_csv('Metadata_Ebooks.csv')

print(f"Original dataset shape: {dataset.shape}")
print(f"Columns: {dataset.columns.tolist()}")

# Remove unwanted categories
categories_to_remove = ['LGBT Studies', 'Religious', 'Romance', 'Youth', 'History', 'Drama', 'Fiction', 'Philosophy']
dataset = dataset[~dataset['Category'].isin(categories_to_remove)]

# Remove low-volume categories
category_counts = dataset['Category'].value_counts()
min_samples = 50
categories_to_keep = category_counts[category_counts >= min_samples].index.tolist()
dataset = dataset[dataset['Category'].isin(categories_to_keep)]
print(f"Kept {len(dataset)} samples from {len(categories_to_keep)} categories")
print(f"Categories kept: {categories_to_keep}")

# Date features
dataset['Published_date'] = pd.to_datetime(dataset['Published'], format='%b-%y')
dataset_creation_date = dataset['Published_date'].max()
dataset['days_since_published'] = (dataset_creation_date - dataset['Published_date']).dt.days

# Title features (keep the most useful ones)
dataset['title_word_count'] = dataset['Title'].str.split().str.len()
dataset['title_length'] = dataset['Title'].str.len()
dataset['title_has_number'] = dataset['Title'].str.contains(r'\d').astype(int)
dataset['title_has_guide'] = dataset['Title'].str.contains('Guide', case=False).astype(int)
dataset['title_has_handbook'] = dataset['Title'].str.contains('Handbook', case=False).astype(int)

# Page features
dataset['pages'] = dataset['Pages']
dataset['pages_log'] = np.log1p(dataset['Pages'])

# Publisher popularity - keep only the most predictive stats
publisher_stats = dataset.groupby('Publisher').agg({
    'Downloads': ['median', 'count']  # Remove mean to avoid correlation
}).reset_index()
publisher_stats.columns = ['Publisher', 'publisher_median_downloads', 'publisher_book_count']

# Add smoothing for publishers with few books
global_median = dataset['Downloads'].median()
publisher_stats['publisher_median_smoothed'] = publisher_stats.apply(
    lambda row: (row['publisher_median_downloads'] * row['publisher_book_count'] + global_median * 3) / (row['publisher_book_count'] + 3),
    axis=1
)

dataset = dataset.merge(publisher_stats, on='Publisher', how='left')

# Time features - keep only the most predictive ones
dataset['days_since_published_squared'] = dataset['days_since_published'] ** 2
# Remove days_since_published_log to reduce correlation

# Month features
dataset['published_month'] = dataset['Published'].str.split('-').str[0]
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
dataset['published_month_num'] = dataset['published_month'].map(month_map)
dataset['published_quarter'] = dataset['published_month_num'].apply(lambda x: (x-1)//3 + 1)

# Drop original columns
dataset = dataset.drop(columns=['Title', 'Publisher', 'Published', 'published_month', 'Published_date'])

# Check for any remaining missing values
print(f"\nMissing values before dropping: {dataset.isnull().sum().sum()}")
dataset = dataset.dropna()
print(f"Dataset shape after dropping NA: {dataset.shape}")

# Prepare features - SELECT ONLY THE BEST ONES
category_dummies = pd.get_dummies(dataset['Category'], prefix='Category', dtype='float')
month_dummies = pd.get_dummies(dataset['published_month_num'], prefix='Month', dtype='float')
quarter_dummies = pd.get_dummies(dataset['published_quarter'], prefix='Quarter', dtype='float')

# Select only the most predictive features
X = pd.concat([
    category_dummies,
    month_dummies,
    quarter_dummies,
    dataset[['title_word_count', 'title_length',
             'days_since_published', 'days_since_published_squared',
             'pages', 'pages_log',
             'publisher_median_smoothed',  # Use smoothed median (less correlated)
             'publisher_book_count']]  # Keep count for context
], axis=1)

# Remove any features that might be redundant
# publisher_median_smoothed already captures publisher popularity

# LOG TRANSFORM TARGET
y = np.log1p(dataset['Downloads'])

print(f"\nFeature count: {X.shape[1]}")
print(f"Sample count: {X.shape[0]}")
print(f"Target range: {y.min():.2f} to {y.max():.2f}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with optimized parameters
regressor = CatBoostRegressor(
    verbose=100,
    random_state=42,
    iterations=300,  # Slightly fewer iterations
    learning_rate=0.05,
    depth=6,  # Reduced depth to prevent overfitting
    l2_leaf_reg=5,  # Increased regularization
    early_stopping_rounds=50
)
regressor.fit(X_train, y_train)

# Predict (on log scale)
y_pred_log = regressor.predict(X_test)

# Convert back to original scale
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

# Evaluate
print(f"\n{'='*50}")
print(f"Model Performance (Original Scale):")
print(f"{'='*50}")
print(f"MAE: {mean_absolute_error(y_test_original, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_original, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test_original, y_pred):.2f}")

# Feature importance (normalized to percentages)
feature_importance = regressor.get_feature_importance()
total_importance = sum(feature_importance)
normalized_importance = [(imp / total_importance * 100) for imp in feature_importance]

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance,
    'Importance_Percent': normalized_importance
}).sort_values('Importance', ascending=False)

print(f"\n{'='*50}")
print("Top 15 Most Important Features:")
print(f"{'='*50}")
print(importance_df.head(15).to_string(index=False))

# ============================================
# Save model and artifacts
# ============================================
print(f"\n{'='*50}")
print("Saving model and artifacts...")
print(f"{'='*50}")

# Save the model
joblib.dump(regressor, 'models/book_download_predictor.joblib')

# Save feature names for validation
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'models/feature_names.joblib')

# Save feature importance for reference
joblib.dump(importance_df, 'models/feature_importance.joblib')

# Save category mappings
available_categories = [col.replace('Category_', '') for col in category_dummies.columns]
joblib.dump(available_categories, 'models/available_categories.joblib')

# Save month mapping
joblib.dump(month_map, 'models/month_mapping.joblib')

# Save publisher stats for reference
publisher_stats_dict = publisher_stats.to_dict('records')
joblib.dump(publisher_stats_dict, 'models/publisher_stats.joblib')

# Save model metadata
model_metadata = {
    'training_date': datetime.now().isoformat(),
    'r2_score': float(r2_score(y_test_original, y_pred)),
    'mae': float(mean_absolute_error(y_test_original, y_pred)),
    'rmse': float(np.sqrt(mean_squared_error(y_test_original, y_pred))),
    'feature_count': len(feature_names),
    'sample_count': len(dataset),
    'categories': available_categories,
    'dataset_creation_date': dataset_creation_date.isoformat()
}
joblib.dump(model_metadata, 'models/model_metadata.joblib')

print("✅ Model and artifacts saved successfully!")
print(f"📁 Files saved in './models/' directory")
print(f"\n📊 Model Metadata:")
print(f"   - R² Score: {model_metadata['r2_score']:.4f}")
print(f"   - MAE: {model_metadata['mae']:.2f}")
print(f"   - RMSE: {model_metadata['rmse']:.2f}")
print(f"   - Features: {model_metadata['feature_count']}")
print(f"   - Samples: {model_metadata['sample_count']}")
print(f"   - Categories: {len(model_metadata['categories'])}")