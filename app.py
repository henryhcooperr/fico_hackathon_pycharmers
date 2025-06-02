#!/usr/bin/env python3
"""
Credit Score Optimizer App - Better than CreditKarma
"""

from credit_score_optimizer import CreditScoreOptimizer
import json

def get_user_input():

    
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
    print("\n Now enter your financial details:")
    
    user_data = {}
    
    prompts = {
        'Annual_Income': "Annual income ($): ",
        'Outstanding_Debt': "Total debt across all accounts ($): ",
        'Credit_Utilization_Ratio': "Credit card utilization (0-1, e.g., 0.3 for 30%): ",
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
        'Payment_Behaviour': 'Low_spent_Small_value_payments'
    })
    
    # Get user inputs
    for feature, prompt in prompts.items():
        while True:
            value = input(prompt).strip()
            try:
                if 'Ratio' in feature:
                    user_data[feature] = float(value)
                elif feature in ['Num_Credit_Card', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries']:
                    user_data[feature] = int(value)
                else:
                    user_data[feature] = float(value)
                break
            except:
                print("Please enter a valid number")
    
    # Calculate monthly salary from annual income
    user_data['Monthly_Inhand_Salary'] = user_data['Annual_Income'] / 12
    
    return current_score, user_data

def display_recommendations(results):
    print(f"📊 YOUR PERSONALIZED CREDIT IMPROVEMENT PLAN"))
    print(f"\nCurrent Credit Score: {results['current_score']}")
    
    # Quick wins
    if results['quick_wins']:
        print("QUICK WINS (Easy changes with immediate impact):")
        for i, rec in enumerate(results['quick_wins'], 1):
            print(f"\n{i}. {rec['specific_action']}")
            print(f"   📈 Expected Improvement: +{rec['predicted_improvement']} points")
    
    # High impact
    if results['high_impact']:
        print("\nHIGH IMPACT ACTIONS (Bigger effort, bigger rewards):")
        for i, rec in enumerate(results['high_impact'], 1):
            if rec not in results['quick_wins']:
                print(f"\n{i}. {rec['specific_action']}")
                print(f"   📈 Expected Improvement: +{rec['predicted_improvement']} points")
    
    # Calculate total potential
    total_potential = sum(rec['predicted_improvement'] for rec in results['recommendations'][:3])
    
    print(f"\n🎯 TOTAL POTENTIAL IMPROVEMENT: +{total_potential} points")
    print(f"   Your score could reach: {results['current_score'] + total_potential}")
    
    # Show priority order
    print("\n📋 COMPLETE ACTION PLAN (in priority order):")
    for i, rec in enumerate(results['recommendations'][:5], 1):
        effort_emoji = "🟢" if rec['effort_score'] <= 3 else "🟡" if rec['effort_score'] <= 6 else "🔴"
        print(f"\n{i}. {effort_emoji} {rec['feature'].replace('_', ' ').title()}")
        print(f"   Change: {rec['current_value']:.1f} → {rec['target_value']:.1f}")
        print(f"   Impact: +{rec['predicted_improvement']} points")

def main():
    """Main application"""
    # Load model
    optimizer = CreditScoreOptimizer()
    
    try:
        optimizer.load_model()
    except:
        print("XXX No trained model found. Please run train_optimizer.py first!")
        return
    
    # Get user input
    current_score, user_data = get_user_input()
    
    # Get recommendations
    print("\nAnalyzing your profile and running simulations...")
    results = optimizer.get_recommendations(user_data, current_score)
    
    # Display results
    display_recommendations(results)
    
    # Save results option
    save = input("\nSave your personalized plan? (y/n): ").lower()
    if save == 'y':
        with open('my_credit_improvement_plan.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Plan saved to 'my_credit_improvement_plan.json'")
    
    print("\n✨ Good luck improving your credit score!")

if __name__ == "__main__":
    main()