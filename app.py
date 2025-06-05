#!/usr/bin/env python3
"""
Credit Score Optimizer App - Better than CreditKarma
"""

from credit_score_optimizer import CreditScoreOptimizer
import json

def get_user_input():
    print("🎯 CREDIT SCORE OPTIMIZER")
    print("Let's analyze your credit and find ways to improve it!\n")
    
    # Get current score
    while True:
        score = input("Your current credit score (300-850): ").strip()
        try:
            current_score = int(score)
            if 300 <= current_score <= 850:
                break
            print("Please enter a score between 300-850")
        except:
            print("Please enter a valid number")
    
    # Get key financial metrics
    print("\nNow enter your financial details:")
    
    user_data = {}
    
    prompts = {
        'Annual_Income': "Annual income ($): ",
        'Outstanding_Debt': "Total debt across all accounts ($): ",
        'Credit_Utilization_Ratio': "Credit card utilization (%, e.g., 30 for 30%): ",
        'Num_Credit_Card': "Number of credit cards: ",
        'Num_of_Delayed_Payment': "Number of late payments (last 2 years): ",
        'Num_Credit_Inquiries': "Credit inquiries (last 2 years): ",
        'Monthly_Balance': "Average monthly bank balance ($): ",
        'Amount_invested_monthly': "Monthly investment amount ($): ",
        'Credit_History_Age': "Age of oldest credit account (years): ",
        'Total_EMI_per_month': "Total monthly debt payments ($): "
    }
    
    # Set defaults for less critical features
    user_data.update({
        'Age': 30,
        'Monthly_Inhand_Salary': 5000,
        'Num_Bank_Accounts': 2,
        'Interest_Rate': 15,
        'Num_of_Loan': 1,
        'Delay_from_due_date': 5,
        'Changed_Credit_Limit': 2,
        'Credit_Mix': 'Standard',
        'Payment_of_Min_Amount': 'No',
        'Payment_Behaviour': 'Low_spent_Small_value_payments',
        'Type_of_Loan': 'Auto Loan, Credit-Builder Loan'
    })
    
    # Get user inputs
    for feature, prompt in prompts.items():
        while True:
            value = input(prompt).strip()
            try:
                if feature == 'Credit_Utilization_Ratio':
                    # Handle percentage input
                    value_float = float(value)
                    # Convert to ratio if entered as percentage
                    if value_float > 1:
                        user_data[feature] = value_float / 100
                    else:
                        user_data[feature] = value_float
                elif feature in ['Num_Credit_Card', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries']:
                    user_data[feature] = int(value)
                elif feature == 'Credit_History_Age':
                    # Convert years to months
                    user_data[feature] = float(value) * 12
                else:
                    user_data[feature] = float(value)
                break
            except:
                print("Please enter a valid number")
    
    # Calculate monthly salary from annual income
    user_data['Monthly_Inhand_Salary'] = user_data['Annual_Income'] / 12
    
    return current_score, user_data

def display_recommendations(results):
    print("\n" + "="*60)
    print("📊 YOUR PERSONALIZED CREDIT IMPROVEMENT PLAN")
    print("="*60)
    
    print(f"\nCurrent Credit Score: {results['current_score']}")
    
    # Get current category if available
    if 'current_category' in results:
        print(f"Current Category: {results['current_category']}")
    
    # Quick wins
    if results['quick_wins']:
        print("\n🎯 QUICK WINS (Easy changes with immediate impact):")
        print("-" * 50)
        for i, rec in enumerate(results['quick_wins'], 1):
            print(f"\n{i}. {rec['specific_action']}")
            print(f"   📈 Expected Improvement: +{rec['predicted_improvement']} points")
            print(f"   ⚡ Effort Level: {rec['effort_score']:.1f}/10 (Easy)")
    
    # High impact
    if results['high_impact']:
        print("\n💪 HIGH IMPACT ACTIONS (Bigger effort, bigger rewards):")
        print("-" * 50)
        for i, rec in enumerate(results['high_impact'], 1):
            if rec not in results['quick_wins']:
                print(f"\n{i}. {rec['specific_action']}")
                print(f"   📈 Expected Improvement: +{rec['predicted_improvement']} points")
                print(f"   ⚡ Effort Level: {rec['effort_score']:.1f}/10")
    
    # Calculate total potential
    total_potential = sum(rec['predicted_improvement'] for rec in results['recommendations'][:3])
    
    print(f"\n🚀 TOTAL POTENTIAL IMPROVEMENT: +{total_potential} points")
    print(f"   Your score could reach: {results['current_score'] + total_potential}")
    
    # Credit score categories
    new_score = results['current_score'] + total_potential
    if new_score >= 800:
        category = "Excellent"
        emoji = "🌟"
    elif new_score >= 740:
        category = "Very Good"
        emoji = "⭐"
    elif new_score >= 670:
        category = "Good"
        emoji = "✅"
    elif new_score >= 580:
        category = "Fair"
        emoji = "✓"
    else:
        category = "Poor"
        emoji = "⚠️"
    
    print(f"   New Category: {emoji} {category}")
    
    # Show priority order
    print("\n📋 COMPLETE ACTION PLAN (in priority order):")
    print("-" * 50)
    for i, rec in enumerate(results['recommendations'][:7], 1):
        effort_emoji = "🟢" if rec['effort_score'] <= 3 else "🟡" if rec['effort_score'] <= 6 else "🔴"
        feature_name = rec['feature'].replace('_', ' ').title()
        
        print(f"\n{i}. {effort_emoji} {feature_name}")
        
        # Handle different value types
        if isinstance(rec['current_value'], (int, float)):
            if 'Ratio' in rec['feature'] or 'Rate' in rec['feature']:
                print(f"   Change: {rec['current_value']:.1%} → {rec['target_value']:.1%}")
            elif 'Num_' in rec['feature'] or rec['feature'] in ['Age', 'Credit_History_Age']:
                print(f"   Change: {int(rec['current_value'])} → {int(rec['target_value'])}")
            else:
                print(f"   Change: ${rec['current_value']:,.0f} → ${rec['target_value']:,.0f}")
        else:
            # For non-numeric values (like "Multiple factors")
            print(f"   Type: {rec['change_amount']}")
        
        print(f"   Impact: +{rec['predicted_improvement']} points")
        print(f"   Effort: {rec['effort_score']:.1f}/10")
    
    # Category probability if available
    if 'category_probabilities' in results:
        print("\n📊 SCORE CATEGORY PROBABILITIES:")
        print("-" * 50)
        for category, prob in results['category_probabilities'].items():
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{category:8} [{bar}] {prob:.1%}")

def save_plan(results, filename='my_credit_improvement_plan.json'):
    """Save the improvement plan to a file"""
    # Create a more readable version of the plan
    plan = {
        'analysis_date': str(pd.Timestamp.now().date()) if 'pd' in globals() else 'Today',
        'current_score': results['current_score'],
        'potential_new_score': results['current_score'] + sum(rec['predicted_improvement'] for rec in results['recommendations'][:3]),
        'quick_wins': [
            {
                'action': rec['specific_action'],
                'improvement': rec['predicted_improvement'],
                'effort': rec['effort_score']
            }
            for rec in results['quick_wins']
        ],
        'all_recommendations': [
            {
                'feature': rec['feature'],
                'action': rec['specific_action'],
                'improvement': rec['predicted_improvement'],
                'effort': rec['effort_score']
            }
            for rec in results['recommendations'][:10]
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(plan, f, indent=2)
    
    return filename

def main():
    """Main application"""
    print("💳 CREDIT SCORE OPTIMIZER - Better than Credit Karma! 💳")
    print("="*60)
    
    # Load model
    optimizer = CreditScoreOptimizer()
    
    try:
        optimizer.load_model()
        print("✅ Model loaded successfully!\n")
    except:
        print("❌ No trained model found. Please run train_optimizer.py first!")
        return
    
    # Get user input
    current_score, user_data = get_user_input()
    
    # Get recommendations
    print("\n🔍 Analyzing your profile and running simulations...")
    print("This may take a few seconds...\n")
    
    try:
        results = optimizer.get_recommendations(user_data, current_score)
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")
        print("Please check your input values and try again.")
        return
    
    # Display results
    display_recommendations(results)
    
    # Save results option
    print("\n" + "="*60)
    save = input("\n💾 Save your personalized plan? (y/n): ").lower()
    if save == 'y':
        try:
            import pandas as pd
        except:
            pass
        filename = save_plan(results)
        print(f"✅ Plan saved to '{filename}'")
        print("📄 You can open this file to review your plan anytime!")
    
    print("\n✨ Good luck improving your credit score!")
    print("💡 Remember: Consistency is key. Small changes add up to big improvements!")
    print("="*60)

if __name__ == "__main__":
    main()