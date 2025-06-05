import pickle
import pandas as pd
import requests
import re

with open("credit_optimizer_model 1.pkl", "rb") as f:
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
    conversation_history.append({"role": "user", "content": prompt})
    url = "http://localhost:8011/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3",
        "messages": conversation_history,
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"].strip()
        conversation_history.append({"role": "assistant", "content": reply})
        return reply
    except Exception:
        return "LLaMA error: unable to connect to server."

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
    "Monthly_Balance": "How much do you usually owe monthly across all accounts?",
    "Annual_Income": "What is your total yearly income before taxes?",
    "Amount_invested_monthly": "How much money do you usually save or invest monthly?"
}

def run_credit_optimizer(user_data):
    print("\nAnalyzing your responses...\n")
    X_input = preprocessor.transform(pd.DataFrame([user_data]))
    prediction = score_model.predict(X_input)[0]
    print(f"Estimated Credit Score: {int(prediction)}")

    print("\nBasic Tips:")
    print(ask_llama(
        f"The user has this credit profile: {user_data}. Their estimated credit score is {int(prediction)}. "
        f"List 3 basic ways they can improve their credit score. Be friendly and clear."
    ))

    more = input("\nWould you like more advanced financial tips? (yes/no): ").strip().lower()
    if more in ["yes", "y"]:
        conversation_history.clear()
        conversation_history.append({
            "role": "system",
            "content": (
                "You are a helpful financial assistant. Based on the user's credit profile and score, give 3 specific advanced personal finance tips. "
                "Do NOT ask the user any questions — just provide guidance."
            )
        })
        print("\nAdvanced Tips:")
        print(ask_llama(
            f"The user has this credit profile: {user_data}. Their estimated score is {int(prediction)}. "
            f"Provide 3 advanced financial strategies tailored to this profile."
        ))
    else:
        print("Thank you. Wishing you financial success!")

    explain_score(user_data)
        
def explain_score(user_data):
    print("\nFeatures Impact on Your Score:")

 
    defaults = {key: 0 for key in user_data}
    X_base = preprocessor.transform(pd.DataFrame([defaults]))
    base_score = score_model.predict(X_base)[0]

    total = 0.0

    for field in user_data:
        modified = defaults.copy()
        modified[field] = user_data[field]
        X_mod = preprocessor.transform(pd.DataFrame([modified]))
        mod_score = score_model.predict(X_mod)[0]

        delta = mod_score - base_score
        total += delta
        direction = "increased" if delta > 0 else "decreased" if delta < 0 else "had no effect on"
        print(f"- {field.replace('_',' ')} {direction} your score by {abs(delta):.1f} points")

    print(f"\nTotal predicted score: {int(base_score + total)}")



def main():
    print("\nWelcome to the Fico Buddy Credit Score Assistant!")
    print("Answer a few friendly questions to get a credit score estimate.")
    print("Type 'skip' to skip or 'exit' to quit.\n")

    user_data = {}
    for field, question in field_questions.items():
        retry = 0
        while retry < 2:
            print(f"\n{question}")
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                return
            if user_input.lower() in ["skip", "s"]:
                break

            val = extract_number(user_input, field_name=field)
            if val is not None:
                user_data[field] = val
                break
            else:
                retry += 1
                if retry == 1:
                    clarification = ask_llama(f"Can you rephrase this question for someone who didn't understand: '{question}'")
                    print(f"\n{clarification}")
                else:
                    print("Skipping this one.")

    if len(user_data) >= 5:
        run_credit_optimizer(user_data)
    else:
        print("Not enough responses to make a prediction.")

if __name__ == "__main__":
    main()
