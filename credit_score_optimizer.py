#!/usr/bin/env python3
"""
Provides specific, actionable recommendations with predicted point improvements
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
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
        print("🚀 Training Credit Score Optimizer...")
        
        # Load data
        df = pd.read_csv(dataset_path)
        
        # IMPORTANT: Split by customer BEFORE dropping Customer_ID
        unique_customers = df['Customer_ID'].unique()
        print(f"Total unique customers: {len(unique_customers)}")
        
        # Split customers 80/20
        from sklearn.model_selection import train_test_split
        train_customers, test_customers = train_test_split(
            unique_customers, 
            test_size=0.2, 
            random_state=42
        )
        
        # Create train/test sets ensuring no customer appears in both
        train_df = df[df['Customer_ID'].isin(train_customers)].copy()
        test_df = df[df['Customer_ID'].isin(test_customers)].copy()
        
        print(f"Train: {len(train_customers)} customers, {len(train_df)} records")
        print(f"Test: {len(test_customers)} customers, {len(test_df)} records")
        
        train_df = self._prepare_training_data(train_df)
        test_df = self._prepare_training_data(test_df)
        
        
        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        
        # Fit preprocessor ONLY on training data
        self.preprocessor.fit(train_df)
        
        # Transform both sets
        X_train = self.preprocessor.transform(train_df)
        X_test = self.preprocessor.transform(test_df)
        
        # Create numeric scores from categories
        score_map = {'Poor': 450, 'Standard': 650, 'Good': 780}
        y_train_numeric = train_df['Credit_Score'].map(score_map)
        y_test_numeric = test_df['Credit_Score'].map(score_map)
        y_train_category = train_df['Credit_Score']
        y_test_category = test_df['Credit_Score']
        
        # Train score prediction model
        print("\nTraining score prediction model...")
        self.score_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.score_model.fit(X_train, y_train_numeric)
        
        # Evaluate regression model
        from sklearn.metrics import mean_absolute_error, r2_score
        train_pred = self.score_model.predict(X_train)
        test_pred = self.score_model.predict(X_test)
        
        print(f"Score Model - Train MAE: {mean_absolute_error(y_train_numeric, train_pred):.2f}")
        print(f"Score Model - Test MAE: {mean_absolute_error(y_test_numeric, test_pred):.2f}")
        print(f"Score Model - Test R²: {r2_score(y_test_numeric, test_pred):.3f}")
        
        # Train category model
        print("\nTraining category classification model...")
        self.category_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.category_model.fit(X_train, y_train_category)
        
        # Evaluate classification model
        from sklearn.metrics import classification_report, accuracy_score
        train_cat_pred = self.category_model.predict(X_train)
        test_cat_pred = self.category_model.predict(X_test)
        
        print(f"Category Model - Train Accuracy: {accuracy_score(y_train_category, train_cat_pred):.3f}")
        print(f"Category Model - Test Accuracy: {accuracy_score(y_test_category, test_cat_pred):.3f}")
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test_category, test_cat_pred))
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.preprocessor.feature_columns,
            'importance': self.score_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10))
        
        # Calculate feature impacts through simulation (use training data)
        print("\nCalculating feature impacts...")
        self._calculate_feature_impacts(X_train, train_df)
        
        print("\nTraining complete!")
        
        # Optional: Return metrics for logging
        return {
            'train_mae': mean_absolute_error(y_train_numeric, train_pred),
            'test_mae': mean_absolute_error(y_test_numeric, test_pred),
            'test_r2': r2_score(y_test_numeric, test_pred),
            'test_accuracy': accuracy_score(y_test_category, test_cat_pred)
        }
        
    def _prepare_training_data(self, df):
        """Prepare training data"""
        # Remove unnecessary columns
        cols_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Handle Age outliers and invalid values
        if 'Age' in df.columns:
            # Convert Age to numeric, treating errors as NaN
            df['Age'] = pd.to_numeric(df['Age'].astype(str).str.replace('_', ''), errors='coerce')
            # Remove obvious outliers (negative ages or unrealistic ages)
            df = df[(df['Age'] > 0) & (df['Age'] < 120)]
            # Fill any remaining NaN ages with median
            if df['Age'].isna().any():
                df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # Remove rows where Credit_Score is missing
        if 'Credit_Score' in df.columns:
            df = df.dropna(subset=['Credit_Score'])
            # Ensure Credit_Score is one of the expected categories
            valid_scores = ['Poor', 'Standard', 'Good']
            df = df[df['Credit_Score'].isin(valid_scores)]
        
        # Basic outlier removal for other numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in df.columns:
                # Remove values that are more than 5 standard deviations from the mean
                mean = df[col].mean()
                std = df[col].std()
                df = df[np.abs(df[col] - mean) <= 5 * std]
        
        print(f"Data shape after cleaning: {df.shape}")
        print(f"Columns in dataset: {list(df.columns)}")
        
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
        
        top_features = self.feature_importance.head(15)
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            
            # Skip if not a modifiable feature
            if feature not in self.preprocessor.numeric_features:
                continue
                
            if feature not in user_data:
                continue
            
            # Generate scenarios
            scenarios = self._generate_scenarios(feature, user_data[feature], user_processed)
            
            for scenario in scenarios:
                # Predict new score
                new_score = int(self.score_model.predict(scenario['data'])[0])
                improvement = new_score - current_score
                
                if improvement > 5:  # Only include meaningful improvements
                    recommendation = {
                        'feature': feature,
                        'current_value': user_data[feature],
                        'target_value': scenario['new_value'],
                        'change_amount': scenario['change_amount'],
                        'predicted_improvement': improvement,
                        'effort_score': scenario['effort'],
                        'specific_action': self._generate_specific_action(
                            feature, user_data[feature], scenario['new_value'], improvement
                        ),
                        'importance': row['importance']
                    }
                    recommendations.append(recommendation)
        
        
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
                user_data, user_processed, current_score, recommendations[:3]
            )
            recommendations.extend(combo_recommendations)
        
        return {
            'current_score': current_score,
            'recommendations': recommendations[:7],  # Top 7 recommendations
            'quick_wins': [r for r in recommendations if r['effort_score'] <= 3][:3],
            'high_impact': [r for r in recommendations if r['predicted_improvement'] >= 50][:3]
        }
    
    def _generate_scenarios(self, feature, current_value, user_processed):
        """Generate what-if scenarios for a feature"""
        scenarios = []
        feature_idx = self.preprocessor.feature_columns.index(feature)
        
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
        
        # Define features where HIGHER is better
        HIGHER_IS_BETTER = [
            'Amount_invested_monthly',
            'Monthly_Balance',
            'Credit_History_Age',  # But can't change this
            'Annual_Income',  # But shouldn't recommend this
            'Monthly_Inhand_Salary'  # But shouldn't recommend this
        ]
        
        # Special cases
        SPECIAL_CASES = ['Num_Credit_Card', 'Num_Bank_Accounts']
        
        # Skip non-actionable features
        NON_ACTIONABLE = ['Annual_Income', 'Monthly_Inhand_Salary', 'Credit_History_Age', 'Age']
        if feature in NON_ACTIONABLE:
            return []
        
        # Special handling for specific features
        if feature == 'Credit_Utilization_Ratio':
            # Always recommend lowering utilization
            if current_value > 0.05:
                targets = [0.3, 0.25, 0.1, 0.05] if current_value > 0.3 else [current_value * 0.5, current_value * 0.25]
                for target in targets:
                    if target < current_value:
                        test_data = user_processed.copy()
                        scaled_target = self.preprocessor.scalers[feature].transform([[target]])[0][0]
                        test_data.iloc[0, feature_idx] = scaled_target
                        
                        effort = self._calculate_effort(feature, current_value, target)
                        scenarios.append({
                            'data': test_data,
                            'new_value': target,
                            'change_amount': target - current_value,
                            'effort': effort
                        })
                        
        elif feature in ['Outstanding_Debt', 'Num_of_Delayed_Payment']:
            # Always recommend reducing these
            reductions = [0.25, 0.5, 0.75, 1.0]
            for reduction in reductions:
                new_value = current_value * (1 - reduction)
                if feature == 'Num_of_Delayed_Payment':
                    new_value = int(new_value)
                    
                test_data = user_processed.copy()
                scaled_value = self.preprocessor.scalers[feature].transform([[new_value]])[0][0]
                test_data.iloc[0, feature_idx] = scaled_value
                
                effort = self._calculate_effort(feature, current_value, new_value)
                scenarios.append({
                    'data': test_data,
                    'new_value': new_value,
                    'change_amount': new_value - current_value,
                    'effort': effort
                })
                
        elif feature == 'Num_Credit_Card':
            # Special logic: only recommend fewer cards if user has many
            if current_value > 5:
                # Too many cards - recommend reduction
                for new_value in [4, 3]:
                    if new_value < current_value:
                        test_data = user_processed.copy()
                        scaled_value = self.preprocessor.scalers[feature].transform([[new_value]])[0][0]
                        test_data.iloc[0, feature_idx] = scaled_value
                        
                        effort = self._calculate_effort(feature, current_value, new_value)
                        scenarios.append({
                            'data': test_data,
                            'new_value': new_value,
                            'change_amount': new_value - current_value,
                            'effort': effort
                        })
            # Don't recommend more cards if already have 3+
            
        elif feature == 'Num_Credit_Inquiries':
            # Can only wait for inquiries to age off (they disappear after 2 years)
            # Don't suggest increasing! Only suggest waiting
            if current_value > 0:
                # Inquiries age off over time
                for new_value in [max(0, current_value - 2), max(0, current_value - 4), 0]:
                    if new_value < current_value:
                        test_data = user_processed.copy()
                        scaled_value = self.preprocessor.scalers[feature].transform([[new_value]])[0][0]
                        test_data.iloc[0, feature_idx] = scaled_value
                        
                        effort = 1  # Easy - just wait
                        scenarios.append({
                            'data': test_data,
                            'new_value': new_value,
                            'change_amount': new_value - current_value,
                            'effort': effort
                        })
                        
        elif feature in LOWER_IS_BETTER:
            # For features that should decrease
            reductions = [0.25, 0.5, 0.75]
            for reduction in reductions:
                new_value = current_value * (1 - reduction)
                test_data = user_processed.copy()
                scaled_value = self.preprocessor.scalers[feature].transform([[new_value]])[0][0]
                test_data.iloc[0, feature_idx] = scaled_value
                
                effort = self._calculate_effort(feature, current_value, new_value)
                scenarios.append({
                    'data': test_data,
                    'new_value': new_value,
                    'change_amount': new_value - current_value,
                    'effort': effort
                })
                
        elif feature in HIGHER_IS_BETTER:
            # For features that should increase
            increases = [0.25, 0.5, 1.0]
            for increase in increases:
                new_value = current_value * (1 + increase)
                test_data = user_processed.copy()
                scaled_value = self.preprocessor.scalers[feature].transform([[new_value]])[0][0]
                test_data.iloc[0, feature_idx] = scaled_value
                
                effort = self._calculate_effort(feature, current_value, new_value)
                scenarios.append({
                    'data': test_data,
                    'new_value': new_value,
                    'change_amount': new_value - current_value,
                    'effort': effort
                })
                
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
            }
        }
        
        template = actions.get(feature, {}).get('template', 
            f"Adjust {feature} from {{current}} to {{target}} for {{improvement}} point improvement.")
        
        return template.format(current=current, target=target, improvement=improvement)
    
    def _generate_combination_scenarios(self, user_data, user_processed, current_score, top_recs):
        """Generate scenarios combining multiple changes"""
        combo_recs = []
        
        # Create a copy of the processed data to apply changes
        test_data = user_processed.copy()
        total_effort = 0
        changes = []
        applied_changes = {}
        
        # Iterate through top recommendations and apply changes
        for rec in top_recs[:3]:
            feature = rec['feature']
            
            # Check if feature exists in the processed data
            if feature not in test_data.columns:
                print(f"Warning: Feature '{feature}' not found in processed data")
                continue
                
            # Store the changes to apply
            applied_changes[feature] = rec['target_value']
            total_effort += rec['effort_score']
            changes.append(f"{feature}: {rec['current_value']:.1f} → {rec['target_value']:.1f}")
        
        # If we have valid changes to apply
        if applied_changes:
            # Create a copy of the original user data for transformation
            modified_user_data = user_data.copy()
            
            # Apply all changes to the original data
            for feature, target_value in applied_changes.items():
                if feature in modified_user_data.columns:
                    modified_user_data[feature] = target_value
            
            # Re-process the entire modified data through the preprocessor
            try:
                test_data_transformed = self.preprocessor.transform(modified_user_data)
                
                # If transform returns numpy array, convert back to DataFrame
                if isinstance(test_data_transformed, np.ndarray):
                    test_data_transformed = pd.DataFrame(
                        test_data_transformed, 
                        columns=user_processed.columns
                    )
            except Exception as e:
                print(f"Error transforming combined data: {e}")
                return combo_recs
            
            try:
                new_score = int(self.score_model.predict(test_data_transformed)[0])
                improvement = new_score - current_score
                
                # Calculate expected improvement (sum of individual improvements)
                expected_improvement = sum(r['predicted_improvement'] for r in top_recs[:len(applied_changes)])
                
                # Only recommend if combination provides significant benefit
                # (more than 80% of sum of individual improvements suggests synergy)
                if improvement > expected_improvement * 0.8:
                    avg_effort = total_effort / len(applied_changes) if applied_changes else 0
                    
                    combo_recs.append({
                        'feature': 'Combined Actions',
                        'current_value': 'Multiple factors',
                        'target_value': 'Optimized values',
                        'change_amount': f'{len(applied_changes)} changes',
                        'predicted_improvement': improvement,
                        'effort_score': avg_effort,
                        'specific_action': f"Combine these actions for maximum impact (+{improvement} points total):\n" + 
                                        "\n".join(f"• {change}" for change in changes),
                        'importance': 1.0,
                        'synergy_bonus': improvement - expected_improvement  # Track how much extra benefit
                    })
                    
            except Exception as e:
                print(f"Error predicting combined score: {e}")
        
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

