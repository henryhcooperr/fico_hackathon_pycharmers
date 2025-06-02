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


# Initial system message for LLaMA
conversation_history = [
    {
        "role": "system",
        "content": (
            "You are a friendly credit assistant. Ask one simple, human question at a time to help estimate a user's credit score. "
            "If the user seems confused, rephrase your question simply. Never assume or answer yourself. Wait for user input."
        )
    }
]


# Call local LLaMA server
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
def extract_number(text):
    text = text.lower()
    if "yes" in text:
        return 1.0
    elif "no" in text:
        return 0.0
    nums = re.findall(r"[\d,.]+", text.replace(",", ""))
    return float(nums[0]) if nums else None


# Questions ordered by credit importance
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


# Run prediction and offer suggestions
def run_credit_optimizer(user_data):
    print("\n Analyzing your responses...\n")
    X_input = preprocessor.transform(pd.DataFrame([user_data]))
    prediction = score_model.predict(X_input)[0]


    print(f" Estimated Credit Score: {int(prediction)}")


    improve_prompt = (
        f"The user has this credit profile: {user_data}. Their estimated credit score is {int(prediction)}. "
        f"Give 3 helpful ways to improve it in a friendly tone."
    )
    print("\n LLaMA Suggestions:\n" + ask_llama(improve_prompt))


    more = input("\nWould you like additional tips to manage money or maintain your score? (yes/no): ").strip().lower()
    if more in ["yes", "y"]:
        advice_prompt = (
            f"The user has this credit profile: {user_data}. Their estimated score is {int(prediction)}. "
            f"Give some bonus personal finance tips, even if the score is good."
        )
        print("\n More Tips:\n" + ask_llama(advice_prompt))
    else:
        print(" Got it! Wishing you financial success!")


# Main Q&A Loop
def main():
    print("\n Welcome to the LLaMA Credit Score Assistant!")
    print("Answer a few friendly questions to get a credit score estimate.")
    print("Type 'skip' to skip or 'exit' to quit.\n")


    user_data = {}
    for field, question in field_questions.items():
        retry = 0
        while retry < 2:
            print(f"\n {question}")
            user_input = input("You: ").strip()


            if user_input.lower() in ["exit", "quit"]:
                print(" Goodbye!")
                return
            if user_input.lower() in ["skip", "s"]:
                break

            val = extract_number(user_input)
            if val is not None:
                user_data[field] = val
                break
            else:
                retry += 1
                if retry == 1:
                    clarification = ask_llama(f"Can you rephrase this question for someone who didn't understand: '{question}'")
                    print(f"\n {clarification}")
                else:
                    print(" Skipping this one.")


    if len(user_data) >= 5:
        run_credit_optimizer(user_data)
    else:
        print("Not enough responses to make a prediction.")


if __name__ == "__main__":
    main()
