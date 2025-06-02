import pickle
import pandas as pd
import numpy as np
import requests
import re

# Load the trained model
with open("credit_optimizer_model.pkl", "rb") as f:
    model_data = pickle.load(f)

preprocessor = model_data["preprocessor"]
score_model = model_data["score_model"]

# 🧠LLaMA memory
conversation_history = [
    {
        "role": "system",
        "content": (
            "You are a friendly credit assistant. Ask one simple, human question at a time to help estimate a user's credit score. "
            "If the user seems confused or says 'what do you mean', rephrase your question to be even easier to understand. "
            "Never assume or answer the questions yourself. Always wait for the user to reply."
        )
    }
]

#  Ask LLaMA with memory
def ask_llama(prompt):
    conversation_history.append({"role": "user", "content": prompt})
    url = "http://localhost:11434/api/chat"  # ← CHANGE THIS LINE
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3",  # Make sure this matches what you pulled
        "messages": conversation_history,
        "stream": False  # ← ADD THIS (Ollama uses different format)
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        reply = response.json()["message"]["content"].strip()  # ← CHANGE THIS TOO
        conversation_history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"🛑 LLaMA error: {e}"
# yes/no
def extract_number(text, field_name=None):
    """Extract numbers with context awareness"""
    text = text.lower().strip()
    
    # Handle field-specific logic
    if field_name == "Credit_Utilization_Ratio":
        # Handle percentage inputs (30%, 30 percent, etc.)
        percent_match = re.search(r'(\d+\.?\d*)\s*(?:%|percent)', text)
        if percent_match:
            return float(percent_match.group(1)) / 100 if float(percent_match.group(1)) > 1 else float(percent_match.group(1))
        # If they just say a number for utilization, assume it's a percentage
        simple_num = re.search(r'^(\d+)$', text.strip())
        if simple_num:
            num = float(simple_num.group(1))
            return num / 100 if num > 1 else num
    
    # For payment/inquiry fields, handle common responses
    if field_name in ["Num_of_Delayed_Payment", "Num_Credit_Inquiries"]:
        if any(word in text for word in ["no", "none", "zero", "never", "0"]):
            return 0.0
        if "once" in text or "one" in text or "1" in text:
            return 1.0
        if "twice" in text or "two" in text or "2" in text:
            return 2.0
        if "few" in text:
            return 3.0  # Assume "few" means 3
    
    # Handle income/debt with k/m notation
    text_clean = text.replace("$", "").replace(",", "")
    
    # Check for thousands (50k, 50K, 50 thousand)
    k_match = re.search(r'(\d+\.?\d*)\s*(?:k|thousand)', text_clean, re.IGNORECASE)
    if k_match:
        return float(k_match.group(1)) * 1000
    
    # Check for millions (1.5m, 2 million)
    m_match = re.search(r'(\d+\.?\d*)\s*(?:m|million)', text_clean, re.IGNORECASE)
    if m_match:
        return float(m_match.group(1)) * 1000000
    
    # Standard number extraction
    numbers = re.findall(r'\d+\.?\d*', text_clean)
    if numbers:
        return float(numbers[0])
    
    return None

#  User questions
field_questions = {
    "Annual_Income": "How much money do you make per year before taxes?",
    "Outstanding_Debt": "How much total money do you owe right now (loans, credit cards, etc)?",
    "Credit_Utilization_Ratio": "Roughly what percent of your credit are you using? Like 10%, 30%, or more?",
    "Num_Credit_Card": "How many credit cards do you currently have?",
    "Num_of_Delayed_Payment": "How many times have you missed a payment?",
    "Num_Credit_Inquiries": "How many times did you apply for credit in the last year?",
    "Monthly_Balance": "How much do you usually owe across all accounts monthly?",
    "Amount_invested_monthly": "How much do you usually save or invest each month?",
    "Credit_History_Age": "How many years have you had credit (any loans or cards)?",
    "Total_EMI_per_month": "How much do you pay monthly on all loans and cards combined?"
}

# Predict + get suggestions
def run_credit_optimizer(user_data):
    print("\n Analyzing your profile...")
    X_input = preprocessor.transform(pd.DataFrame([user_data]))
    prediction = score_model.predict(X_input)[0]

    print(f"\n💳 Estimated Credit Score: {int(prediction)}")

    improvement_prompt = (
        f"The user has this credit profile: {user_data}. Their estimated credit score is {int(prediction)}. "
        "Give 3 helpful ways they could improve their score without sounding harsh."
    )
    print("\n LLaMA Suggestions:\n" + ask_llama(improvement_prompt))

    #  more tips
    extra_tips = input("\n Would you like more personalized tips to manage your money or keep your score strong? (yes/no): ").strip().lower()
    if extra_tips in ["yes", "y"]:
        advice_prompt = (
            f"The user has this credit profile: {user_data}. Their estimated score is {int(prediction)}. "
            "Give them a few practical, positive money tips — even if their credit score is already good."
        )
        print("\n Extra Advice from LLaMA:\n" + ask_llama(advice_prompt))
    else:
        print("\n No problem! Wishing you financial success!")

# Main loop
def main():
    print("Welcome to the LLaMA Credit Score Assistant!")
    print("I'll ask you some easy questions to estimate your credit score.")
    print("Type 'skip' to skip or 'exit' to quit.\n")

    user_data = {}
    asked_fields = []

    for field, prompt in field_questions.items():
        retry_count = 0
        while retry_count < 2:
            response = ask_llama(prompt)
            print(f"\nLLaMA: {response}")
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                return
            if user_input.lower() in ["skip", "s"]:
                break

            val = extract_number(user_input, field_name=field)
            if val is not None:
                user_data[field] = val
                asked_fields.append(field)
                break
            else:
                retry_count += 1
                print("That didn’t look like a number. I’ll try rephrasing...")
        else:
            print("Moving on to the next question...\n")

    if len(user_data) >= 5:
        run_credit_optimizer(user_data)
    else:
        print("Not enough answers to make a prediction.")

if __name__ == "__main__":
    main()
