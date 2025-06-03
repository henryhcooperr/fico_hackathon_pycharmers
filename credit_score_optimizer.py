#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
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
        df_clean = self._clean_data(df)
        
        # Identify categorical features
        exclude_cols = ['Credit_Score', 'Customer_ID', 'Name', 'SSN', 'ID', 'Month']
        for col in df_clean.columns:
            if col not in exclude_cols and col not in self.numeric_features:
                try:
                    pd.to_numeric(df_clean[col], errors='raise')
                except:
                    self.categorical_features.append(col)
        
        # Fit encoders
        for feature in self.categorical_features:
            if feature in df_clean.columns:
                self.encoders[feature] = LabelEncoder()
                df_clean[feature] = df_clean[feature].fillna('Unknown')
                self.encoders[feature].fit(df_clean[feature].astype(str))
        
        # Fit scalers
        for feature in self.numeric_features:
            if feature in df_clean.columns:
                self.scalers[feature] = StandardScaler()
                df_clean[feature] = pd.to_numeric(df_clean[feature], errors='coerce')
                df_clean[feature] = df_clean[feature].fillna(df_clean[feature].median())
                self.scalers[feature].fit(df_clean[[feature]])
        
        self.feature_columns = [col for col in df_clean.columns if col not in exclude_cols]
        
    def transform(self, data):
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        df_clean = self._clean_data(df)
        
        # Transform categorical
        for feature in self.categorical_features:
            if feature in df_clean.columns:
                df_clean[feature] = df_clean[feature].fillna('Unknown').astype(str)
                try:
                    df_clean[feature] = self.encoders[feature].transform(df_clean[feature])
                except ValueError:
                    # Handle unseen categories
                    known_categories = set(self.encoders[feature].classes_)
                    df_clean[feature] = df_clean[feature].apply(
                        lambda x: x if x in known_categories else self.encoders[feature].classes_[0]
                    )
                    df_clean[feature] = self.encoders[feature].transform(df_clean[feature])
        
        # Transform numeric
        for feature in self.numeric_features:
            if feature in df_clean.columns:
                df_clean[feature] = pd.to_numeric(df_clean[feature], errors='coerce')
                if df_clean[feature].isna().any():
                    median_val = self.scalers[feature].mean_[0]
                    df_clean[feature] = df_clean[feature].fillna(median_val)
                df_clean[feature] = self.scalers[feature].transform(df_clean[[feature]])
        
        # Ensure all features present
        for col in self.feature_columns:
            if col not in df_clean.columns:
                df_clean[col] = 0
                
        return df_clean[self.feature_columns]
    
    def _clean_data(self, df):
        df_clean = df.copy()
        
        # Convert Credit_History_Age from string format to months
        if 'Credit_History_Age' in df_clean.columns:
            def parse_credit_history(value):
                if pd.isna(value):
                    return np.nan
                try:
                    value = str(value)
                    if 'Year' in value and 'Month' in value:
                        parts = value.split(' and ')
                        years = int(parts[0].split()[0])
                        months = int(parts[1].split()[0])
                        return years * 12 + months
                    elif 'Year' in value:
                        return int(value.split()[0]) * 12
                    elif 'Month' in value:
                        return int(value.split()[0])
                    else:
                        return float(value)
                except:
                    return np.nan
            
            df_clean['Credit_History_Age'] = df_clean['Credit_History_Age'].apply(parse_credit_history)
        
        # Convert percentages to ratios
        if 'Credit_Utilization_Ratio' in df_clean.columns:
            if df_clean['Credit_Utilization_Ratio'].max() > 1:
                df_clean['Credit_Utilization_Ratio'] = df_clean['Credit_Utilization_Ratio'] / 100
        
        # Handle string numerics
        for col in self.numeric_features:
            if col in df_clean.columns and col != 'Credit_History_Age':
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = pd.to_numeric(
                    df_clean[col].str.replace('_', '').str.strip(), 
                    errors='coerce'
                )
        
        # Fill missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df_clean[col] = df_clean[col].fillna(median_val)
        
        categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            df_clean[col] = df_clean[col].astype(str)
            
        return df_clean


class CreditScoreOptimizer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.score_model = None
        self.category_model = None
        self.feature_importance = None
        self.feature_impacts = {}
        
    def train(self, dataset_path):
        df = pd.read_csv(dataset_path)
        df = df.drop_duplicates()
        
        # Split by customer to prevent data leakage
        unique_customers = df['Customer_ID'].unique()
        train_customers, temp_customers = train_test_split(unique_customers, test_size=0.3, random_state=42)
        val_customers, test_customers = train_test_split(temp_customers, test_size=0.5, random_state=42)
        
        train_df = df[df['Customer_ID'].isin(train_customers)].copy()
        val_df = df[df['Customer_ID'].isin(val_customers)].copy()
        test_df = df[df['Customer_ID'].isin(test_customers)].copy()
        
        # Prepare data
        train_df = self._prepare_training_data(train_df)
        val_df = self._prepare_training_data(val_df)
        test_df = self._prepare_training_data(test_df)
        
        # Fit preprocessor on training data only
        self.preprocessor.fit(train_df)
        
        # Transform all sets
        X_train = self.preprocessor.transform(train_df)
        X_val = self.preprocessor.transform(val_df)
        X_test = self.preprocessor.transform(test_df)
        
        # Create numeric scores
        score_map = {'Poor': 450, 'Standard': 650, 'Good': 780}
        y_train_numeric = train_df['Credit_Score'].map(score_map)
        y_val_numeric = val_df['Credit_Score'].map(score_map)
        y_test_numeric = test_df['Credit_Score'].map(score_map)
        y_train_category = train_df['Credit_Score']
        y_test_category = test_df['Credit_Score']
        
        # Train XGBoost regression model
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
        
        xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror', eval_metric='rmse')
        xgb_search = RandomizedSearchCV(xgb_model, xgb_params, n_iter=30, cv=5, 
                                      scoring='neg_mean_absolute_error', n_jobs=-1, 
                                      random_state=42, verbose=0)
        xgb_search.fit(X_train, y_train_numeric)
        
        # Train final model
        self.score_model = xgb.XGBRegressor(**xgb_search.best_params_, random_state=42, 
                                          objective='reg:squarederror', eval_metric='rmse')
        self.score_model.fit(X_train, y_train_numeric)
        
        # Train XGBoost classifier
        category_map = {'Poor': 0, 'Standard': 1, 'Good': 2}
        y_train_cat_numeric = y_train_category.map(category_map)
        y_test_cat_numeric = y_test_category.map(category_map)
        
        xgb_clf = xgb.XGBClassifier(random_state=42, objective='multi:softprob', 
                                    num_class=3, eval_metric='mlogloss')
        clf_params = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 150]
        }
        
        self.category_model = xgb.XGBClassifier(**{'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 100},
                                              random_state=42, objective='multi:softprob', 
                                              num_class=3, eval_metric='mlogloss')
        self.category_model.fit(X_train, y_train_cat_numeric)
        
        # Calculate feature importance
        importances = self.score_model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': self.preprocessor.feature_columns,
            'importance': importances / importances.sum()
        }).sort_values('importance', ascending=False)
        
        # Calculate feature impacts
        self._calculate_feature_impacts(X_train, train_df)
        
        # Evaluate
        test_pred = self.score_model.predict(X_test)
        test_mae = mean_absolute_error(y_test_numeric, test_pred)
        test_r2 = r2_score(y_test_numeric, test_pred)
        
        # Evaluate classifier
        test_cat_pred = self.category_model.predict(X_test)
        inv_category_map = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        test_cat_pred_labels = pd.Series(test_cat_pred).map(inv_category_map)
        test_accuracy = accuracy_score(y_test_category, test_cat_pred_labels)
        
        print(f"\nTest Accuracy: {test_accuracy:.3f}")
        
        print("\nTop 10 Most Important Features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        print("\nFeature Impacts (10% change):")
        for feature, impact_data in sorted(self.feature_impacts.items(), 
                                         key=lambda x: abs(x[1]['avg_impact_per_10pct']), 
                                         reverse=True)[:10]:
            avg_impact = impact_data['avg_impact_per_10pct']
            print(f"{feature}: {avg_impact:+.1f} points")
        
        return {'test_mae': test_mae, 'test_r2': test_r2, 'test_accuracy': test_accuracy}
        
    def _prepare_training_data(self, df):
        # Remove unnecessary columns
        cols_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Clean Credit_Score
        df = df.dropna(subset=['Credit_Score'])
        
        # Handle different Credit_Score formats
        if df['Credit_Score'].dtype in ['int64', 'float64', 'int32', 'float32']:
            if set(df['Credit_Score'].unique()) <= {0, 1, 2}:
                score_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
                df['Credit_Score'] = df['Credit_Score'].map(score_mapping)
        else:
            df['Credit_Score'] = df['Credit_Score'].astype(str).str.strip().str.title()
            score_mapping = {
                'Poor': 'Poor', 'Bad': 'Poor', 'Low': 'Poor',
                'Standard': 'Standard', 'Average': 'Standard', 'Medium': 'Standard', 'Fair': 'Standard',
                'Good': 'Good', 'High': 'Good', 'Excellent': 'Good', 'Very Good': 'Good'
            }
            df['Credit_Score'] = df['Credit_Score'].map(lambda x: score_mapping.get(x, x))
        
        # Keep only valid scores
        valid_scores = ['Poor', 'Standard', 'Good']
        df = df[df['Credit_Score'].isin(valid_scores)]
        
        # Clean Age
        if 'Age' in df.columns:
            df['Age'] = pd.to_numeric(df['Age'].astype(str).str.replace('_', ''), errors='coerce')
            df = df[(df['Age'] >= 0) & (df['Age'] <= 150)]
            df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # Cap extreme values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())
            
            if col == 'Credit_Utilization_Ratio':
                df.loc[df[col] > 10, col] = 10
                df.loc[df[col] < 0, col] = 0
            elif col in ['Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Num_Credit_Card', 
                        'Num_Bank_Accounts', 'Num_of_Loan']:
                df.loc[df[col] < 0, col] = 0
                df.loc[df[col] > 100, col] = 100
                df[col] = df[col].round()
            elif col in ['Outstanding_Debt', 'Annual_Income', 'Monthly_Balance', 
                        'Monthly_Inhand_Salary', 'Total_EMI_per_month', 'Amount_invested_monthly']:
                df.loc[df[col] < 0, col] = 0
                cap_value = df[col].quantile(0.99) * 2
                df.loc[df[col] > cap_value, col] = cap_value
            elif col == 'Interest_Rate':
                df.loc[df[col] < 0, col] = 0
                df.loc[df[col] > 50, col] = 50
            elif col == 'Credit_History_Age':
                df.loc[df[col] < 0, col] = 0
                df.loc[df[col] > 600, col] = 600
        
        return df.drop_duplicates()
    
    def _calculate_feature_impacts(self, X, original_df):
        sample_size = min(1000, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        
        for feature in self.preprocessor.numeric_features:
            if feature not in self.preprocessor.feature_columns:
                continue
                
            feature_idx = self.preprocessor.feature_columns.index(feature)
            impacts = []
            
            for idx in sample_indices[:100]:
                try:
                    base_score = self.score_model.predict(X.iloc[idx:idx+1])[0]
                    test_data = X.iloc[idx:idx+1].copy()
                    current_val = test_data.iloc[0, feature_idx]
                    
                    if pd.isna(current_val) or current_val == 0:
                        continue
                    
                    # Determine improvement direction
                    if feature in ['Outstanding_Debt', 'Num_of_Delayed_Payment', 
                                  'Delay_from_due_date', 'Num_Credit_Inquiries']:
                        new_val = current_val * 0.9
                    else:
                        new_val = current_val * 1.1
                    
                    test_data.iloc[0, feature_idx] = new_val
                    new_score = self.score_model.predict(test_data)[0]
                    impact = new_score - base_score
                    
                    if not pd.isna(impact):
                        impacts.append(impact)
                except:
                    continue
            
            if impacts:
                self.feature_impacts[feature] = {
                    'avg_impact_per_10pct': float(np.mean(impacts)),
                    'std_impact': float(np.std(impacts))
                }
    
    def get_recommendations(self, user_data, current_score=None):
        user_processed = self.preprocessor.transform(user_data)
        
        if current_score is None:
            current_score = int(self.score_model.predict(user_processed)[0])
        
        recommendations = []
        top_features = self.feature_importance.head(15)
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            
            if feature not in self.preprocessor.numeric_features or feature not in user_data:
                continue
            
            scenarios = self._generate_scenarios(feature, user_data[feature], user_data, current_score)
            
            for scenario in scenarios:
                if scenario['predicted_improvement'] > 5:
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
        
        # Keep best recommendation per feature
        best_by_feature = {}
        for rec in recommendations:
            feature = rec['feature']
            if feature not in best_by_feature or rec['predicted_improvement'] > best_by_feature[feature]['predicted_improvement']:
                best_by_feature[feature] = rec
        
        recommendations = list(best_by_feature.values())
        
        # Add combination recommendations
        if len(recommendations) >= 2:
            combo_recommendations = self._generate_combination_scenarios(
                user_data, current_score, recommendations[:3]
            )
            recommendations.extend(combo_recommendations)
        
        # Sort by impact/effort ratio
        recommendations.sort(
            key=lambda x: x['predicted_improvement'] / (x['effort_score'] + 1), 
            reverse=True
        )
        
        return {
            'current_score': current_score,
            'recommendations': recommendations[:7],
            'quick_wins': [r for r in recommendations if r['effort_score'] <= 3][:3],
            'high_impact': [r for r in recommendations if r['predicted_improvement'] >= 50][:3]
        }
    
    def _generate_scenarios(self, feature, current_value, user_data, current_score):
        scenarios = []
        
        LOWER_IS_BETTER = [
            'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Num_of_Delayed_Payment',
            'Delay_from_due_date', 'Num_Credit_Inquiries', 'Interest_Rate', 'Total_EMI_per_month'
        ]
        
        NON_ACTIONABLE = ['Annual_Income', 'Monthly_Inhand_Salary', 'Credit_History_Age', 'Age']
        if feature in NON_ACTIONABLE:
            return []
        
        # Generate test values
        test_values = []
        
        if feature == 'Credit_Utilization_Ratio' and current_value > 0.05:
            targets = [0.3, 0.25, 0.1, 0.05] if current_value > 0.3 else [current_value * 0.5, current_value * 0.25]
            test_values = [t for t in targets if t < current_value]
        elif feature == 'Num_of_Delayed_Payment' and current_value > 0:
            test_values = [max(0, current_value - 1), max(0, current_value - 2), 0]
        elif feature == 'Outstanding_Debt' and current_value > 0:
            test_values = [current_value * 0.75, current_value * 0.5, current_value * 0.25]
        elif feature == 'Num_Credit_Card' and current_value > 5:
            test_values = [4, 3]
        elif feature == 'Num_Credit_Inquiries' and current_value > 0:
            test_values = [max(0, current_value - 2), max(0, current_value - 4), 0]
        elif feature in LOWER_IS_BETTER and current_value > 0:
            test_values = [current_value * 0.75, current_value * 0.5, current_value * 0.25]
        else:
            test_values = [current_value * 1.25, current_value * 1.5, current_value * 2.0]
        
        # Test each value
        for new_value in test_values:
            if new_value == current_value:
                continue
                
            modified_data = user_data.copy()
            modified_data[feature] = new_value
            
            try:
                modified_processed = self.preprocessor.transform(modified_data)
                new_score = int(self.score_model.predict(modified_processed)[0])
                improvement = new_score - current_score
                
                if improvement > 0:
                    effort = self._calculate_effort(feature, current_value, new_value)
                    scenarios.append({
                        'new_value': new_value,
                        'change_amount': new_value - current_value,
                        'predicted_improvement': improvement,
                        'effort': effort
                    })
            except:
                continue
                
        return scenarios
    
    def _calculate_effort(self, feature, current_value, new_value):
        change_pct = abs(new_value - current_value) / (current_value + 1)
        
        effort_map = {
            'Credit_Utilization_Ratio': 3 + change_pct * 5,
            'Num_of_Delayed_Payment': 2,
            'Outstanding_Debt': 5 + change_pct * 10,
            'Num_Credit_Inquiries': 1,
            'Amount_invested_monthly': 4 + change_pct * 6,
            'Monthly_Balance': 4 + change_pct * 6,
            'Credit_History_Age': 10,
            'Interest_Rate': 6,
            'Total_EMI_per_month': 5 + change_pct * 7,
            'Num_Credit_Card': 3,
        }
        
        base_effort = effort_map.get(feature, 5)
        return min(10, base_effort)
    
    def _generate_specific_action(self, feature, current, target, improvement):
        actions = {
            'Credit_Utilization_Ratio': 
                f"Pay down credit cards to reduce utilization from {current:.1%} to {target:.1%}. "
                f"If you have multiple cards, pay down the highest utilization card first. "
                f"This could improve your score by {improvement} points.",
            
            'Outstanding_Debt': 
                f"Reduce total debt from ${current:,.0f} to ${target:,.0f}. "
                f"Start with highest interest debt first. "
                f"Potential score improvement: {improvement} points.",
            
            'Num_of_Delayed_Payment': 
                f"Reduce late payments from {current:.0f} to {target:.0f}. "
                f"Set up autopay for all bills immediately. "
                f"This change alone could boost your score by {improvement} points.",
            
            'Num_Credit_Inquiries': 
                f"Stop applying for new credit. Let inquiries age from {current:.0f} to {target:.0f}. "
                f"Wait 6 months before any new applications. "
                f"Expected improvement: {improvement} points.",
            
            'Amount_invested_monthly': 
                f"Increase monthly investments from ${current:,.0f} to ${target:,.0f}. "
                f"Start with employer 401k match if available. "
                f"Predicted score increase: {improvement} points.",
            
            'Monthly_Balance': 
                f"Build savings from ${current:,.0f} to ${target:,.0f} monthly balance. "
                f"Set up automatic transfer right after payday. "
                f"This demonstrates stability and could add {improvement} points.",
            
            'Total_EMI_per_month': 
                f"Reduce monthly debt payments from ${current:,.0f} to ${target:,.0f}. "
                f"Consider consolidating high-interest debts. "
                f"Could improve score by {improvement} points.",
            
            'Num_Credit_Card': 
                f"Reduce number of credit cards from {current:.0f} to {target:.0f}. "
                f"Close newer cards with lower limits first. "
                f"This simplification could add {improvement} points.",
        }
        
        return actions.get(feature, 
            f"Adjust {feature} from {current} to {target} for {improvement} point improvement.")
    
    def _generate_combination_scenarios(self, user_data, current_score, top_recs):
        combo_recs = []
        
        if len(top_recs) < 2:
            return combo_recs
        
        modified_data = user_data.copy()
        total_effort = 0
        changes = []
        
        for rec in top_recs[:3]:
            feature = rec['feature']
            if feature in modified_data:
                modified_data[feature] = rec['target_value']
                total_effort += rec['effort_score']
                changes.append(f"{feature}: {rec['current_value']:.1f} → {rec['target_value']:.1f}")
        
        try:
            modified_processed = self.preprocessor.transform(modified_data)
            new_score = int(self.score_model.predict(modified_processed)[0])
            improvement = new_score - current_score
            
            expected_improvement = sum(r['predicted_improvement'] for r in top_recs[:3])
            
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
        except:
            pass
        
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
    
    def load_model(self, filepath='credit_optimizer_model.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.preprocessor = data['preprocessor']
            self.score_model = data['score_model']
            self.category_model = data['category_model']
            self.feature_importance = data['feature_importance']
            self.feature_impacts = data['feature_impacts']