import pickle
import pandas as pd
import requests
import re
import json

with open("credit_optimizer_model.pkl", "rb") as f:
    model_data = pickle.load(f)

preprocessor = model_data["preprocessor"]
score_model = model_data["score_model"]

# system prompt for LLaMA
conversation_history = [
    {
        "role": "system",
        "content": (
            "You are a friendly credit assistant. Ask one simple, human question at a time to help estimate a user's credit score. "
            "If the user seems confused, rephrase your question simply. Never assume or answer yourself. Wait for user input."
        )
    }
]

def ask_llama(prompt):
    """Fixed version using the generate endpoint which is more reliable"""
    
    # Build conversation context
    full_prompt = ""
    
    # Add conversation history to prompt
    for msg in conversation_history[-5:]:  # Last 5 messages for context
        if msg["role"] == "system":
            full_prompt += f"System: {msg['content']}\n\n"
        elif msg["role"] == "user":
            full_prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            full_prompt += f"Assistant: {msg['content']}\n"
    
    # Add current prompt
    full_prompt += f"User: {prompt}\nAssistant: "
    
    # Use the generate endpoint instead of chat
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3:latest",
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 500
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Extract the generated text
        if "response" in result:
            reply = result["response"].strip()
        else:
            reply = "I couldn't understand the response format."
            
        # Update conversation history
        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": reply})
        
        return reply
        
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it's running (ollama serve)."
    except requests.exceptions.HTTPError as e:
        return f"HTTP Error: {e}. The generate endpoint should work with llama3:latest."
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Try a shorter prompt or check if Ollama is responding."
    except Exception as e:
        return f"Error: {str(e)}"

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                print("✅ Ollama is running. Available models:")
                for model in models:
                    print(f"   - {model['name']}")
                return True
            else:
                print("⚠️  Ollama is running but no models are installed.")
                print("   Run: ollama pull llama3.2")
                return False
        return False
    except:
        print("❌ Cannot connect to Ollama. Make sure it's running:")
        print("   Run: ollama serve")
        return False

def extract_number(text, field_name=None):
    text = text.lower().strip()

    if field_name == "Credit_Utilization_Ratio":
        percent_match = re.search(r'(\d+\.?\d*)\s*(?:%|percent)', text)
        if percent_match:
            val = float(percent_match.group(1))
            return val / 100 if val > 1 else val
        if re.fullmatch(r'\d+', text):
            val = float(text)
            return val / 100 if val > 1 else val

    if field_name in ["Num_of_Delayed_Payment", "Num_Credit_Inquiries"]:
        if any(word in text for word in ["no", "none", "zero", "never", "0"]):
            return 0.0
        if "once" in text or "one" in text or "1" in text:
            return 1.0
        if "twice" in text or "two" in text or "2" in text:
            return 2.0
        if "few" in text:
            return 3.0

    text_clean = text.replace("$", "").replace(",", "")
    k_match = re.search(r'(\d+\.?\d*)\s*(?:k|thousand)', text_clean, re.IGNORECASE)
    if k_match:
        return float(k_match.group(1)) * 1000
    m_match = re.search(r'(\d+\.?\d*)\s*(?:m|million)', text_clean, re.IGNORECASE)
    if m_match:
        return float(m_match.group(1)) * 1000000
    numbers = re.findall(r'\d+\.?\d*', text_clean)
    if numbers:
        return float(numbers[0])
    return None

field_questions = {
    "Credit_Utilization_Ratio": "What percent of your available credit are you currently using? (e.g., 10%, 30%)",
    "Num_of_Delayed_Payment": "Roughly how many times have you missed a payment in the past?",
    "Credit_History_Age": "How many years have you had any form of credit like loans or credit cards?",
    "Outstanding_Debt": "About how much total money do you currently owe (credit cards, loans, etc)?",
    "Total_EMI_per_month": "How much do you pay monthly on all loans and credit cards combined?",
    "Num_Credit_Inquiries": "How many times have you applied for credit in the last year?",
    "Num_Credit_Card": "How many credit cards do you actively use?",
    "Monthly_Balance": "How much do you usually have in your bank account on average?",
    "Annual_Income": "What is your total yearly income before taxes?",
    "Amount_invested_monthly": "How much money do you usually save or invest monthly?"
}

def run_credit_optimizer(user_data):
    print("\n🔍 Analyzing your responses...\n")
    
    # Fill in missing values with defaults
    defaults = {
        'Age': 30,
        'Monthly_Inhand_Salary': user_data.get('Annual_Income', 50000) / 12,
        'Num_Bank_Accounts': 2,
        'Interest_Rate': 15,
        'Num_of_Loan': 1,
        'Delay_from_due_date': 5,
        'Changed_Credit_Limit': 2,
        'Credit_Mix': 'Standard',
        'Payment_of_Min_Amount': 'No',
        'Payment_Behaviour': 'Low_spent_Small_value_payments',
        'Type_of_Loan': 'Auto Loan, Credit-Builder Loan'
    }
    
    # Merge user data with defaults
    complete_data = {**defaults, **user_data}
    
    # Predict score
    X_input = preprocessor.transform(pd.DataFrame([complete_data]))
    prediction = score_model.predict(X_input)[0]
    
    print(f"📊 Estimated Credit Score: {int(prediction)}")
    
    # Determine credit category
    if prediction < 580:
        category = "Poor"
        emoji = "🔴"
    elif prediction < 670:
        category = "Fair"
        emoji = "🟡"
    elif prediction < 740:
        category = "Good"
        emoji = "🟢"
    elif prediction < 800:
        category = "Very Good"
        emoji = "💚"
    else:
        category = "Excellent"
        emoji = "🌟"
    
    print(f"   Category: {emoji} {category}\n")

    print("💡 Basic Tips:")
    tips_prompt = (
        f"The user has a credit score of {int(prediction)} ({category}). "
        f"Their credit utilization is {user_data.get('Credit_Utilization_Ratio', 0)*100:.0f}%, "
        f"they've had {user_data.get('Num_of_Delayed_Payment', 0)} late payments, "
        f"and owe ${user_data.get('Outstanding_Debt', 0):,.0f}. "
        f"Give 3 specific, actionable tips to improve their credit score. Be friendly and encouraging."
    )
    print(ask_llama(tips_prompt))

    more = input("\n🎯 Would you like more advanced financial tips? (yes/no): ").strip().lower()
    if more in ["yes", "y"]:
        # Reset conversation for focused advice
        conversation_history.clear()
        conversation_history.append({
            "role": "system",
            "content": "You are an expert financial advisor. Provide specific, advanced strategies based on the user's profile."
        })
        
        print("\n🚀 Advanced Strategies:")
        advanced_prompt = (
            f"Based on this profile: Score {int(prediction)}, "
            f"Income ${user_data.get('Annual_Income', 0):,}/year, "
            f"Debt ${user_data.get('Outstanding_Debt', 0):,}, "
            f"Monthly payments ${user_data.get('Total_EMI_per_month', 0):,}, "
            f"{user_data.get('Num_Credit_Card', 0)} credit cards. "
            f"Provide 3 advanced financial strategies like debt consolidation, balance transfers, "
            f"credit limit optimization, or investment strategies."
        )
        print(ask_llama(advanced_prompt))
    
    print("\n" + "="*60)
    explain_score(user_data, complete_data, prediction)
        
def explain_score(user_data, complete_data, prediction):
    print("📈 How Each Factor Affects Your Score:\n")

    # Create baseline with all zeros/defaults
    baseline = {key: 0 for key in complete_data}
    baseline.update({
        'Credit_Mix': 'Standard',
        'Payment_of_Min_Amount': 'No',
        'Payment_Behaviour': 'Low_spent_Small_value_payments',
        'Type_of_Loan': 'No Data'
    })
    
    X_base = preprocessor.transform(pd.DataFrame([baseline]))
    base_score = score_model.predict(X_base)[0]

    impacts = []
    
    for field in user_data:
        if field in ['Annual_Income', 'Credit_Utilization_Ratio', 'Num_of_Delayed_Payment', 
                    'Outstanding_Debt', 'Num_Credit_Inquiries', 'Total_EMI_per_month']:
            modified = baseline.copy()
            modified[field] = complete_data[field]
            if field == 'Annual_Income':
                modified['Monthly_Inhand_Salary'] = complete_data[field] / 12
                
            X_mod = preprocessor.transform(pd.DataFrame([modified]))
            mod_score = score_model.predict(X_mod)[0]
            
            delta = mod_score - base_score
            impacts.append((field, delta))
    
    # Sort by absolute impact
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for field, delta in impacts:
        if abs(delta) > 0.1:
            if delta > 0:
                print(f"✅ {field.replace('_',' ').title()}: +{delta:.1f} points")
            else:
                print(f"❌ {field.replace('_',' ').title()}: {delta:.1f} points")
    
    print(f"\n📊 Total Score Breakdown:")
    print(f"   Base Score: {int(base_score)}")
    print(f"   Your Factors: {int(sum(i[1] for i in impacts))}")
    print(f"   Final Score: {int(prediction)}")

def main():
    print("\n🎯 Welcome to the Fico Buddy Credit Score Assistant!")
    print("="*50)
    
    # Test Ollama connection first
    if not test_ollama_connection():
        print("\n⚠️  Continuing without AI assistance...")
        print("You'll still get your credit score estimate!\n")
    
    print("\nAnswer a few friendly questions to get a credit score estimate.")
    print("Type 'skip' to skip or 'exit' to quit.\n")

    user_data = {}
    for field, question in field_questions.items():
        retry = 0
        while retry < 2:
            print(f"\n❓ {question}")
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye! Thanks for using Fico Buddy!")
                return
            if user_input.lower() in ["skip", "s"]:
                break

            val = extract_number(user_input, field_name=field)
            if val is not None:
                user_data[field] = val
                print(f"✓ Got it: {val}")
                break
            else:
                retry += 1
                if retry == 1:
                    print("🤔 I didn't quite catch that. Could you enter a number?")
                    print(f"   (Examples: 30%, $5000, 3 times)")
                else:
                    print("   Skipping this question.")

    if len(user_data) >= 5:
        run_credit_optimizer(user_data)
    else:
        print("\n⚠️  Not enough responses to make an accurate prediction.")
        print("Please answer at least 5 questions for a credit score estimate.")

if __name__ == "__main__":
    main()