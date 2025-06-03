#!/usr/bin/env python3
"""
Provides specific, actionable recommendations with predicted point improvements
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Separate preprocessing pipeline for consistent data handling"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = None
        self.categorical_features = []
        self.numeric_features = [
            'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
            'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
            'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
            'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'
        ]
        
    def fit(self, df):
        """Fit the preprocessor on training data"""
        # Clean data
        df_clean = self._clean_data(df)
        
        # Identify categorical features automatically
        # Exclude target and ID columns
        exclude_cols = ['Credit_Score', 'Customer_ID', 'Name', 'SSN', 'ID', 'Month']
        
        # Find all non-numeric columns that aren't in exclude list
        for col in df_clean.columns:
            if col not in exclude_cols and col not in self.numeric_features:
                # Check if it's truly categorical (not a numeric column stored as string)
                try:
                    pd.to_numeric(df_clean[col], errors='raise')
                except:
                    # It's categorical
                    self.categorical_features.append(col)
        
        print(f"Detected categorical features: {self.categorical_features}")
        
        # Fit encoders for categorical features
        for feature in self.categorical_features:
            if feature in df_clean.columns:
                self.encoders[feature] = LabelEncoder()
                # Handle missing values before encoding
                df_clean[feature] = df_clean[feature].fillna('Unknown')
                self.encoders[feature].fit(df_clean[feature].astype(str))
        
        # Fit scalers for numeric features
        for feature in self.numeric_features:
            if feature in df_clean.columns:
                self.scalers[feature] = StandardScaler()
                # Ensure numeric type
                df_clean[feature] = pd.to_numeric(df_clean[feature], errors='coerce')
                df_clean[feature] = df_clean[feature].fillna(df_clean[feature].median())
                self.scalers[feature].fit(df_clean[[feature]])
        
        # Store feature columns
        self.feature_columns = [col for col in df_clean.columns 
                               if col not in exclude_cols]
        
    def transform(self, data):
        """Transform data using fitted preprocessor"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Clean
        df_clean = self._clean_data(df)
        
        # Transform categorical
        for feature in self.categorical_features:
            if feature in df_clean.columns:
                # Handle missing values
                df_clean[feature] = df_clean[feature].fillna('Unknown')
                # Handle unseen categories
                df_clean[feature] = df_clean[feature].astype(str)
                # Use transform with error handling
                try:
                    df_clean[feature] = self.encoders[feature].transform(df_clean[feature])
                except ValueError:
                    # Handle unseen categories by mapping to most frequent
                    known_categories = set(self.encoders[feature].classes_)
                    df_clean[feature] = df_clean[feature].apply(
                        lambda x: x if x in known_categories else self.encoders[feature].classes_[0]
                    )
                    df_clean[feature] = self.encoders[feature].transform(df_clean[feature])
        
        # Transform numeric
        for feature in self.numeric_features:
            if feature in df_clean.columns:
                # Ensure numeric
                df_clean[feature] = pd.to_numeric(df_clean[feature], errors='coerce')
                # Fill missing
                if df_clean[feature].isna().any():
                    median_val = self.scalers[feature].mean_[0]  # Use fitted mean
                    df_clean[feature] = df_clean[feature].fillna(median_val)
                # Scale
                df_clean[feature] = self.scalers[feature].transform(df_clean[[feature]])
        
        # Ensure all features present
        for col in self.feature_columns:
            if col not in df_clean.columns:
                df_clean[col] = 0
                
        return df_clean[self.feature_columns]
    
    def inverse_transform_numeric(self, feature_name, value):
        """Convert scaled value back to original scale"""
        if feature_name in self.scalers:
            return self.scalers[feature_name].inverse_transform([[value]])[0][0]
        return value
    
    def _clean_data(self, df):
        """Clean the data"""
        df_clean = df.copy()
        
        # Handle string numerics
        for col in self.numeric_features:
            if col in df_clean.columns:
                # First convert to string to handle mixed types
                df_clean[col] = df_clean[col].astype(str)
                # Remove underscores and convert
                df_clean[col] = pd.to_numeric(
                    df_clean[col].str.replace('_', '').str.strip(), 
                    errors='coerce'
                )
        
        # Fill missing values for numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Handle categorical columns (will be handled in transform)
        # Just ensure they're strings
        categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            df_clean[col] = df_clean[col].astype(str)
            
        return df_clean


class CreditScoreOptimizer:
    """Main system for credit score optimization with what-if analysis"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.score_model = None  # Predicts credit score
        self.category_model = None  # Predicts credit category
        self.feature_importance = None
        self.score_ranges = {
            'Poor': (300, 579),
            'Standard': (580, 739),
            'Good': (740, 850)
        }
        self.feature_impacts = {}
        
    def train(self, dataset_path):
        """Train with customer-based split to prevent data leakage"""
        print("🚀 Training Credit Score Optimizer with XGBoost...")
        
        # Load data
        df = pd.read_csv(dataset_path)
        print(f"\n📊 Loaded dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Inspect Credit_Score column
        if 'Credit_Score' in df.columns:
            print(f"\nCredit_Score value counts:")
            print(df['Credit_Score'].value_counts())
            print(f"Credit_Score unique values: {df['Credit_Score'].unique()}")
            print(f"Credit_Score data type: {df['Credit_Score'].dtype}")
            print(f"Sample Credit_Score values: {df['Credit_Score'].head(10).tolist()}")
        else:
            print("\n⚠️ WARNING: 'Credit_Score' column not found!")
            print(f"Available columns: {list(df.columns)}")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"\nRemoved {initial_rows - len(df)} duplicate rows")
        
        # IMPORTANT: Split by customer BEFORE dropping Customer_ID
        unique_customers = df['Customer_ID'].unique()
        print(f"Total unique customers: {len(unique_customers)}")
        
        # Split customers 70/15/15 for train/val/test
        train_customers, temp_customers = train_test_split(
            unique_customers, 
            test_size=0.3, 
            random_state=42
        )
        val_customers, test_customers = train_test_split(
            temp_customers,
            test_size=0.5,
            random_state=42
        )
        
        # Create train/val/test sets ensuring no customer appears in multiple sets
        train_df = df[df['Customer_ID'].isin(train_customers)].copy()
        val_df = df[df['Customer_ID'].isin(val_customers)].copy()
        test_df = df[df['Customer_ID'].isin(test_customers)].copy()
        
        print(f"Train: {len(train_customers)} customers, {len(train_df)} records")
        print(f"Val: {len(val_customers)} customers, {len(val_df)} records")
        print(f"Test: {len(test_customers)} customers, {len(test_df)} records")
        
        # Prepare data - pass the dataset path for diagnostics
        self.dataset_path = dataset_path
        train_df = self._prepare_training_data(train_df.copy())
        val_df = self._prepare_training_data(val_df.copy())
        test_df = self._prepare_training_data(test_df.copy())
        
        # Fit preprocessor ONLY on training data
        self.preprocessor.fit(train_df)
        
        # Transform all sets
        X_train = self.preprocessor.transform(train_df)
        X_val = self.preprocessor.transform(val_df)
        X_test = self.preprocessor.transform(test_df)
        
        # Create numeric scores from categories
        score_map = {'Poor': 450, 'Standard': 650, 'Good': 780}
        y_train_numeric = train_df['Credit_Score'].map(score_map)
        y_val_numeric = val_df['Credit_Score'].map(score_map)
        y_test_numeric = test_df['Credit_Score'].map(score_map)
        y_train_category = train_df['Credit_Score']
        y_val_category = val_df['Credit_Score']
        y_test_category = test_df['Credit_Score']
        
        # Train XGBoost model with hyperparameter tuning
        print("\nTraining XGBoost score prediction model with hyperparameter tuning...")
        
        # XGBoost parameters for strong regularization
        xgb_params = {
            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [3, 5, 7],
            'gamma': [0.1, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.7, 0.8],
            'reg_alpha': [0.01, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2],
            'n_estimators': [100, 150],
            'learning_rate': [0.01, 0.05, 0.1]
        }
        
        xgb_model = xgb.XGBRegressor(
            random_state=42, 
            objective='reg:squarederror',
            eval_metric='rmse'
        )
        
        # Use RandomizedSearchCV for efficiency
        xgb_search = RandomizedSearchCV(
            xgb_model, 
            xgb_params, 
            n_iter=30,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42,
            verbose=0  # Reduce verbosity
        )
        
        # Fit with cross-validation (no early stopping during CV)
        xgb_search.fit(X_train, y_train_numeric)
        
        # Get best parameters
        best_params = xgb_search.best_params_
        print(f"\nBest XGBoost parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Create final model without early stopping to avoid version issues
        print("\nTraining final model...")
        self.score_model = xgb.XGBRegressor(
            **best_params,
            random_state=42,
            objective='reg:squarederror',
            eval_metric='rmse'
        )
        
        # Fit final model
        self.score_model.fit(X_train, y_train_numeric)
        
        # Evaluate regression model on all sets
        train_pred = self.score_model.predict(X_train)
        val_pred = self.score_model.predict(X_val)
        test_pred = self.score_model.predict(X_test)
        
        print("\n📊 Score Model Performance:")
        print(f"Train MAE: {mean_absolute_error(y_train_numeric, train_pred):.2f}")
        print(f"Val MAE: {mean_absolute_error(y_val_numeric, val_pred):.2f}")
        print(f"Test MAE: {mean_absolute_error(y_test_numeric, test_pred):.2f}")
        print(f"Train R²: {r2_score(y_train_numeric, train_pred):.3f}")
        print(f"Val R²: {r2_score(y_val_numeric, val_pred):.3f}")
        print(f"Test R²: {r2_score(y_test_numeric, test_pred):.3f}")
        
        # Train XGBoost classifier for categories
        print("\n🎯 Training XGBoost category classification model...")
        
        xgb_clf_params = {
            'max_depth': [3, 4, 5],
            'min_child_weight': [3, 5],
            'gamma': [0.1, 0.2],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
            'n_estimators': [100, 150],
            'learning_rate': [0.05, 0.1]
        }
        
        # Map categories to numeric for XGBoost classifier
        category_map = {'Poor': 0, 'Standard': 1, 'Good': 2}
        y_train_cat_numeric = y_train_category.map(category_map)
        y_val_cat_numeric = y_val_category.map(category_map)
        y_test_cat_numeric = y_test_category.map(category_map)
        
        xgb_clf = xgb.XGBClassifier(
            random_state=42,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss'
        )
        
        clf_search = GridSearchCV(
            xgb_clf,
            xgb_clf_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0  # Reduce verbosity
        )
        
        clf_search.fit(X_train, y_train_cat_numeric)
        
        # Get best parameters
        best_clf_params = clf_search.best_params_
        print(f"\nBest classifier parameters:")
        for param, value in best_clf_params.items():
            print(f"  {param}: {value}")
        
        # Create final classifier without early stopping to avoid version issues
        print("Training final classifier...")
        self.category_model = xgb.XGBClassifier(
            **best_clf_params,
            random_state=42,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss'
        )
        
        # Fit final classifier
        self.category_model.fit(X_train, y_train_cat_numeric)
        
        # Evaluate classification model
        train_cat_pred = self.category_model.predict(X_train)
        val_cat_pred = self.category_model.predict(X_val)
        test_cat_pred = self.category_model.predict(X_test)
        
        # Convert predictions back to original labels
        inv_category_map = {v: k for k, v in category_map.items()}
        train_cat_pred_labels = pd.Series(train_cat_pred).map(inv_category_map)
        val_cat_pred_labels = pd.Series(val_cat_pred).map(inv_category_map)
        test_cat_pred_labels = pd.Series(test_cat_pred).map(inv_category_map)
        
        print("\n📈 Category Model Performance:")
        print(f"Train Accuracy: {accuracy_score(y_train_category, train_cat_pred_labels):.3f}")
        print(f"Val Accuracy: {accuracy_score(y_val_category, val_cat_pred_labels):.3f}")
        print(f"Test Accuracy: {accuracy_score(y_test_category, test_cat_pred_labels):.3f}")
        
        # Check for overfitting
        train_acc = accuracy_score(y_train_category, train_cat_pred_labels)
        test_acc = accuracy_score(y_test_category, test_cat_pred_labels)
        
        if abs(train_acc - test_acc) > 0.1:
            print("\n⚠️ WARNING: Significant difference between train and test accuracy!")
            print("Consider increasing regularization parameters.")
        else:
            print("\n✅ Model shows good generalization (train/test accuracy difference < 10%)")
        
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test_category, test_cat_pred_labels))
        
        # Calculate feature importance from XGBoost
        importance_dict = self.score_model.get_booster().get_score(importance_type='gain')
        # Convert to array matching feature order
        importances = []
        for i in range(len(self.preprocessor.feature_columns)):
            importances.append(importance_dict.get(f'f{i}', 0))
        importances = np.array(importances)
        importances = importances / importances.sum()
        
        self.feature_importance = pd.DataFrame({
            'feature': self.preprocessor.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n🏆 Top 10 Most Important Features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        # Cross-validation check
        print("\n🔄 Performing cross-validation...")
        cv_scores = cross_val_score(
            self.score_model, 
            X_train, 
            y_train_numeric,
            cv=5,
            scoring='neg_mean_absolute_error'
        )
        print(f"CV MAE scores: {-cv_scores}")
        print(f"CV MAE mean: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Calculate feature impacts
        print("\n📊 Calculating feature impacts...")
        self._calculate_feature_impacts(X_train, train_df)
        
        print("\n✅ Training complete!")
        
        return {
            'train_mae': mean_absolute_error(y_train_numeric, train_pred),
            'val_mae': mean_absolute_error(y_val_numeric, val_pred),
            'test_mae': mean_absolute_error(y_test_numeric, test_pred),
            'train_r2': r2_score(y_train_numeric, train_pred),
            'val_r2': r2_score(y_val_numeric, val_pred),
            'test_r2': r2_score(y_test_numeric, test_pred),
            'train_accuracy': train_acc,
            'val_accuracy': accuracy_score(y_val_category, val_cat_pred_labels),
            'test_accuracy': test_acc,
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std()
        }
        
    def _prepare_training_data(self, df):
        """Prepare training data with better validation"""
        print(f"\n📊 Preparing data - Initial shape: {df.shape}")
        
        # Remove unnecessary columns
        cols_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        print(f"After removing ID columns: {df.shape}")
        
        # Remove rows where Credit_Score is missing FIRST
        if 'Credit_Score' in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=['Credit_Score'])
            print(f"Removed {initial_len - len(df)} rows with missing Credit_Score")
            
            # Check what credit score values we have
            print(f"Unique Credit_Score values: {df['Credit_Score'].unique()[:20]}")  # Show first 20
            
            # Handle different Credit_Score formats
            # First, check if it's numeric (0, 1, 2 or similar)
            if df['Credit_Score'].dtype in ['int64', 'float64', 'int32', 'float32']:
                print("Detected numeric Credit_Score values, converting...")
                # Common numeric mappings
                if set(df['Credit_Score'].unique()) <= {0, 1, 2}:
                    score_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
                elif set(df['Credit_Score'].unique()) <= {1, 2, 3}:
                    score_mapping = {1: 'Poor', 2: 'Standard', 3: 'Good'}
                else:
                    # Use percentiles for other numeric ranges
                    q33 = df['Credit_Score'].quantile(0.33)
                    q66 = df['Credit_Score'].quantile(0.66)
                    df['Credit_Score'] = pd.cut(df['Credit_Score'], 
                                               bins=[-np.inf, q33, q66, np.inf],
                                               labels=['Poor', 'Standard', 'Good'])
                    score_mapping = None
                
                if score_mapping:
                    df['Credit_Score'] = df['Credit_Score'].map(score_mapping)
            else:
                # Handle string values
                df['Credit_Score'] = df['Credit_Score'].astype(str).str.strip().str.title()
                
                # Map any variations to standard values
                score_mapping = {
                    'Poor': 'Poor',
                    'Bad': 'Poor',
                    'Low': 'Poor',
                    'Standard': 'Standard',
                    'Average': 'Standard',
                    'Medium': 'Standard',
                    'Fair': 'Standard',
                    'Good': 'Good',
                    'High': 'Good',
                    'Excellent': 'Good',
                    'Very Good': 'Good'
                }
                
                df['Credit_Score'] = df['Credit_Score'].map(lambda x: score_mapping.get(x, x))
            
            # Now filter for valid scores
            valid_scores = ['Poor', 'Standard', 'Good']
            initial_len = len(df)
            df = df[df['Credit_Score'].isin(valid_scores)]
            
            print(f"Removed {initial_len - len(df)} rows with Credit_Score not in {valid_scores}")
            
            if len(df) == 0:
                print("⚠️ WARNING: All data removed after Credit_Score filtering!")
                print("Attempting to check original data values...")
                # Try to reload and check
                if hasattr(self, 'dataset_path'):
                    try:
                        df_temp = pd.read_csv(self.dataset_path)
                        if 'Credit_Score' in df_temp.columns:
                            print(f"Original Credit_Score values:\n{df_temp['Credit_Score'].value_counts()}")
                            print(f"First 10 Credit_Score values: {df_temp['Credit_Score'].head(10).tolist()}")
                    except:
                        pass
                raise ValueError(
                    "No valid Credit_Score values found. Expected: Poor, Standard, Good. "
                    "Run diagnose_data.py to check your data format."
                )
        
        print(f"After Credit_Score validation: {df.shape}")
        
        # Handle Age outliers and invalid values
        if 'Age' in df.columns and len(df) > 0:
            # Convert Age to numeric, treating errors as NaN
            df['Age'] = pd.to_numeric(df['Age'].astype(str).str.replace('_', ''), errors='coerce')
            initial_len = len(df)
            
            # Very lenient age filtering - just remove impossible values
            df = df[(df['Age'] >= 0) & (df['Age'] <= 150)]
            
            # For any still invalid ages, replace with median instead of removing
            if df['Age'].isna().any():
                median_age = df['Age'].median()
                if pd.isna(median_age):
                    median_age = 35  # Default if all ages are NaN
                df['Age'] = df['Age'].fillna(median_age)
            
            if initial_len - len(df) > 0:
                print(f"Removed {initial_len - len(df)} rows with impossible Age values")
        
        print(f"After Age cleaning: {df.shape}")
        
        # MINIMAL data cleaning - preserve as much data as possible
        if len(df) > 0:
            print("Applying minimal data cleaning to preserve samples...")
            
            # Only fix obviously wrong values without removing rows
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in df.columns:
                    # Replace infinite values with NaN, then with median
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    if df[col].isna().any():
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                    
                    # Cap extreme values instead of removing
                    if col == 'Credit_Utilization_Ratio':
                        # Cap at 10 instead of removing
                        df.loc[df[col] > 10, col] = 10
                        df.loc[df[col] < 0, col] = 0
                    
                    elif col in ['Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Num_Credit_Card', 
                                'Num_Bank_Accounts', 'Num_of_Loan']:
                        # Cap counts at reasonable maximum
                        df.loc[df[col] < 0, col] = 0
                        df.loc[df[col] > 100, col] = 100
                        df[col] = df[col].round()
                    
                    elif col in ['Outstanding_Debt', 'Annual_Income', 'Monthly_Balance', 
                                'Monthly_Inhand_Salary', 'Total_EMI_per_month', 'Amount_invested_monthly']:
                        # Ensure non-negative for money values
                        df.loc[df[col] < 0, col] = 0
                        # Cap at 99th percentile to handle extreme values
                        cap_value = df[col].quantile(0.99) * 2  # 2x the 99th percentile
                        df.loc[df[col] > cap_value, col] = cap_value
                    
                    elif col == 'Interest_Rate':
                        # Interest rates should be reasonable
                        df.loc[df[col] < 0, col] = 0
                        df.loc[df[col] > 50, col] = 50
                    
                    elif col == 'Credit_History_Age':
                        # Credit history in months/years - cap at reasonable max
                        df.loc[df[col] < 0, col] = 0
                        df.loc[df[col] > 600, col] = 600  # 50 years
            
            print(f"After value capping (no rows removed): {df.shape}")
        
        print(f"After outlier removal: {df.shape}")
        
        # Remove any remaining duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        if initial_len - len(df) > 0:
            print(f"Removed {initial_len - len(df)} duplicate rows")
        
        # Ensure we have enough data
        if len(df) < 100:
            print(f"⚠️ WARNING: Only {len(df)} samples after cleaning. Model may not perform well.")
            # If we have too little data, reload and be less aggressive
            if len(df) < 10:
                print("🚨 CRITICAL: Too few samples. Attempting less aggressive cleaning...")
                # This is a fallback - you might need to adjust based on your actual data
                return df  # Return whatever we have
        
        print(f"Final data shape after cleaning: {df.shape}")
        
        if len(df.columns) > 0:
            print(f"Columns in dataset: {list(df.columns)[:10]}...")  # Show first 10
        
        if 'Credit_Score' in df.columns and len(df) > 0:
            print(f"Credit Score distribution:\n{df['Credit_Score'].value_counts()}")
        
        return df
    
    def _calculate_feature_impacts(self, X, original_df):
        """Calculate how much each feature change impacts score"""
        # Sample data for impact analysis
        sample_size = min(1000, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        
        for feature in self.preprocessor.numeric_features:
            if feature not in self.preprocessor.feature_columns:
                continue
                
            feature_idx = self.preprocessor.feature_columns.index(feature)
            impacts = []
            
            # Test different change percentages
            for idx in sample_indices:
                base_score = self.score_model.predict(X.iloc[idx:idx+1])[0]
                
                # Try 10% improvement
                test_data = X.iloc[idx:idx+1].copy()
                current_val = test_data.iloc[0, feature_idx]
                
                # Determine improvement direction
                if feature in ['Outstanding_Debt', 'Credit_Utilization_Ratio', 'Num_of_Delayed_Payment', 
                              'Delay_from_due_date', 'Num_Credit_Inquiries']:
                    # Lower is better
                    new_val = current_val * 0.9
                else:
                    # Higher is better
                    new_val = current_val * 1.1
                    
                test_data.iloc[0, feature_idx] = new_val
                new_score = self.score_model.predict(test_data)[0]
                
                impact = new_score - base_score
                impacts.append(impact)
            
            self.feature_impacts[feature] = {
                'avg_impact_per_10pct': np.mean(impacts),
                'std_impact': np.std(impacts)
            }
    
    def get_recommendations(self, user_data, current_score=None):
        """Get specific recommendations with predicted improvements"""
        # Preprocess user data
        user_processed = self.preprocessor.transform(user_data)
        
        # Predict current score if not provided
        if current_score is None:
            current_score = int(self.score_model.predict(user_processed)[0])

        # Run what-if scenarios
        recommendations = []
        
        # Get top features by importance
        top_features = self.feature_importance.head(15)
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            
            # Skip if not a modifiable numeric feature
            if feature not in self.preprocessor.numeric_features:
                continue
                
            if feature not in user_data:
                continue
            
            # Generate scenarios for this feature
            scenarios = self._generate_scenarios(feature, user_data[feature], user_data, current_score)
            
            for scenario in scenarios:
                if scenario['predicted_improvement'] > 5:  # Only include meaningful improvements
                    recommendation = {
                        'feature': feature,
                        'current_value': user_data[feature],
                        'target_value': scenario['new_value'],
                        'change_amount': scenario['change_amount'],
                        'predicted_improvement': scenario['predicted_improvement'],
                        'effort_score': scenario['effort'],
                        'specific_action': self._generate_specific_action(
                            feature, user_data[feature], scenario['new_value'], scenario['predicted_improvement']
                        ),
                        'importance': row['importance']
                    }
                    recommendations.append(recommendation)
        
        # Keep only the best recommendation per feature
        best_by_feature = {}
        for rec in recommendations:
            feature = rec['feature']
            if feature not in best_by_feature or rec['predicted_improvement'] > best_by_feature[feature]['predicted_improvement']:
                best_by_feature[feature] = rec
        
        recommendations = list(best_by_feature.values())
        
        # Sort by impact/effort ratio
        recommendations.sort(
            key=lambda x: x['predicted_improvement'] / (x['effort_score'] + 1), 
            reverse=True
        )

        # Add combination scenarios
        if len(recommendations) >= 2:
            combo_recommendations = self._generate_combination_scenarios(
                user_data, current_score, recommendations[:3]
            )
            recommendations.extend(combo_recommendations)
        
        # Final sort
        recommendations.sort(
            key=lambda x: x['predicted_improvement'] / (x['effort_score'] + 1), 
            reverse=True
        )
        
        return {
            'current_score': current_score,
            'recommendations': recommendations[:7],  # Top 7 recommendations
            'quick_wins': [r for r in recommendations if r['effort_score'] <= 3][:3],
            'high_impact': [r for r in recommendations if r['predicted_improvement'] >= 50][:3]
        }
    
    def _generate_scenarios(self, feature, current_value, user_data, current_score):
        """Generate what-if scenarios for a feature"""
        scenarios = []
        
        # Define features where LOWER is better
        LOWER_IS_BETTER = [
            'Outstanding_Debt',
            'Credit_Utilization_Ratio', 
            'Num_of_Delayed_Payment',
            'Delay_from_due_date',
            'Num_Credit_Inquiries',
            'Interest_Rate',
            'Total_EMI_per_month'
        ]
        
        # Skip non-actionable features
        NON_ACTIONABLE = ['Annual_Income', 'Monthly_Inhand_Salary', 'Credit_History_Age', 'Age']
        if feature in NON_ACTIONABLE:
            return []
        
        # Generate test values based on feature type
        test_values = []
        
        if feature == 'Credit_Utilization_Ratio':
            # Always recommend lowering utilization
            if current_value > 0.05:
                targets = [0.3, 0.25, 0.1, 0.05] if current_value > 0.3 else [current_value * 0.5, current_value * 0.25]
                test_values = [t for t in targets if t < current_value]
                
        elif feature == 'Num_of_Delayed_Payment':
            # Always recommend reducing these
            if current_value > 0:
                test_values = [max(0, current_value - 1), max(0, current_value - 2), 0]
                
        elif feature == 'Outstanding_Debt':
            # Recommend reducing debt
            if current_value > 0:
                test_values = [current_value * 0.75, current_value * 0.5, current_value * 0.25]
                
        elif feature == 'Num_Credit_Card':
            # Special logic: only recommend fewer cards if user has many
            if current_value > 5:
                test_values = [4, 3]
            # Don't recommend more cards
            
        elif feature == 'Num_Credit_Inquiries':
            # Can only wait for inquiries to age off
            if current_value > 0:
                test_values = [max(0, current_value - 2), max(0, current_value - 4), 0]
                
        elif feature in LOWER_IS_BETTER:
            # For features that should decrease
            if current_value > 0:
                test_values = [current_value * 0.75, current_value * 0.5, current_value * 0.25]
        else:
            # For features that should increase (e.g., savings, investments)
            test_values = [current_value * 1.25, current_value * 1.5, current_value * 2.0]
        
        # Test each value
        for new_value in test_values:
            if new_value == current_value:
                continue
                
            # Create a copy of user data with the modified value
            modified_data = user_data.copy()
            modified_data[feature] = new_value
            
            # Transform the modified data
            try:
                modified_processed = self.preprocessor.transform(modified_data)
                new_score = int(self.score_model.predict(modified_processed)[0])
                improvement = new_score - current_score
                
                if improvement > 0:  # Only add if it improves the score
                    effort = self._calculate_effort(feature, current_value, new_value)
                    scenarios.append({
                        'new_value': new_value,
                        'change_amount': new_value - current_value,
                        'predicted_improvement': improvement,
                        'effort': effort
                    })
            except Exception as e:
                print(f"Error generating scenario for {feature}: {e}")
                continue
                
        return scenarios
    
    def _calculate_effort(self, feature, current_value, new_value):
        """Calculate effort score (1-10) for making a change"""
        change_pct = abs(new_value - current_value) / (current_value + 1)
        
        effort_map = {
            'Credit_Utilization_Ratio': 3 + change_pct * 5,  # Relatively easy
            'Num_of_Delayed_Payment': 2,  # Easy - just pay on time
            'Outstanding_Debt': 5 + change_pct * 10,  # Harder if large amount
            'Num_Credit_Inquiries': 1,  # Very easy - just stop applying
            'Amount_invested_monthly': 4 + change_pct * 6,
            'Monthly_Balance': 4 + change_pct * 6,
            'Credit_History_Age': 10,  # Very hard - requires time
            'Interest_Rate': 6,  # Medium - requires refinancing
            'Total_EMI_per_month': 5 + change_pct * 7,
            'Num_Credit_Card': 3,  # Easy to close cards
        }
        
        base_effort = effort_map.get(feature, 5)
        return min(10, base_effort)
    
    def _generate_specific_action(self, feature, current, target, improvement):
        """Generate specific, actionable advice"""
        actions = {
            'Credit_Utilization_Ratio': {
                'template': "Pay down credit cards to reduce utilization from {current:.1%} to {target:.1%}. "
                           "If you have multiple cards, pay down the highest utilization card first. "
                           "This could improve your score by {improvement} points.",
            },
            'Outstanding_Debt': {
                'template': "Reduce total debt from ${current:,.0f} to ${target:,.0f}. "
                           "Start with highest interest debt first. "
                           "Potential score improvement: {improvement} points.",
            },
            'Num_of_Delayed_Payment': {
                'template': "Reduce late payments from {current:.0f} to {target:.0f}. "
                           "Set up autopay for all bills immediately. "
                           "This change alone could boost your score by {improvement} points.",
            },
            'Num_Credit_Inquiries': {
                'template': "Stop applying for new credit. Let inquiries age from {current:.0f} to {target:.0f}. "
                           "Wait 6 months before any new applications. "
                           "Expected improvement: {improvement} points.",
            },
            'Amount_invested_monthly': {
                'template': "Increase monthly investments from ${current:,.0f} to ${target:,.0f}. "
                           "Start with employer 401k match if available. "
                           "Predicted score increase: {improvement} points.",
            },
            'Monthly_Balance': {
                'template': "Build savings from ${current:,.0f} to ${target:,.0f} monthly balance. "
                           "Set up automatic transfer right after payday. "
                           "This demonstrates stability and could add {improvement} points.",
            },
            'Total_EMI_per_month': {
                'template': "Reduce monthly debt payments from ${current:,.0f} to ${target:,.0f}. "
                           "Consider consolidating high-interest debts. "
                           "Could improve score by {improvement} points.",
            },
            'Num_Credit_Card': {
                'template': "Reduce number of credit cards from {current:.0f} to {target:.0f}. "
                           "Close newer cards with lower limits first. "
                           "This simplification could add {improvement} points.",
            }
        }
        
        template = actions.get(feature, {}).get('template', 
            f"Adjust {feature} from {{current}} to {{target}} for {{improvement}} point improvement.")
        
        return template.format(current=current, target=target, improvement=improvement)
    
    def _generate_combination_scenarios(self, user_data, current_score, top_recs):
        """Generate scenarios combining multiple changes"""
        combo_recs = []
        
        if len(top_recs) < 2:
            return combo_recs
        
        # Apply top 3 recommendations together
        modified_data = user_data.copy()
        total_effort = 0
        changes = []
        
        for rec in top_recs[:3]:
            feature = rec['feature']
            if feature in modified_data:
                modified_data[feature] = rec['target_value']
                total_effort += rec['effort_score']
                changes.append(f"{feature}: {rec['current_value']:.1f} → {rec['target_value']:.1f}")
        
        # Transform and predict
        try:
            modified_processed = self.preprocessor.transform(modified_data)
            new_score = int(self.score_model.predict(modified_processed)[0])
            improvement = new_score - current_score
            
            # Calculate expected improvement (sum of individual improvements)
            expected_improvement = sum(r['predicted_improvement'] for r in top_recs[:3])
            
            # Only recommend if combination provides significant benefit
            if improvement > expected_improvement * 0.8 and improvement > 10:
                avg_effort = total_effort / 3
                
                combo_recs.append({
                    'feature': 'Combined Actions',
                    'current_value': 'Multiple factors',
                    'target_value': 'Optimized values',
                    'change_amount': f'{len(changes)} changes',
                    'predicted_improvement': improvement,
                    'effort_score': avg_effort,
                    'specific_action': f"Combine these actions for maximum impact (+{improvement} points total):\n" + 
                                    "\n".join(f"• {change}" for change in changes),
                    'importance': 1.0,
                    'synergy_bonus': improvement - expected_improvement
                })
                
        except Exception as e:
            print(f"Error in combination scenario: {e}")
        
        return combo_recs
    
    def save_model(self, filepath='credit_optimizer_model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'preprocessor': self.preprocessor,
                'score_model': self.score_model,
                'category_model': self.category_model,
                'feature_importance': self.feature_importance,
                'feature_impacts': self.feature_impacts
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='credit_optimizer_model.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.preprocessor = data['preprocessor']
            self.score_model = data['score_model']
            self.category_model = data['category_model']
            self.feature_importance = data['feature_importance']
            self.feature_impacts = data['feature_impacts']
        print(f"Model loaded from {filepath}")