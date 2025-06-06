from credit_score_optimizer import DataPreprocessor, CreditScoreOptimizer
import pickle
import pickle
import pandas as pd
import requests
import re
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import time
import json
from collections import OrderedDict

with open("credit_optimizer_model2.pkl", "rb") as f:
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
            "Always aim for numeric answers (like amounts, percentages, vounts, or years). Neer skip details."
        )
    }
]

def ask_llama(prompt):
    conversation_history.append({"role": "user", "content": str(prompt)})
    safe_history = [
        {"role": msg["role"], "content": str(msg.get("content", ""))}
        for msg in conversation_history
    ]

    url = "http://localhost:8011/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3",
        "messages": safe_history,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        json_response = response.json()

        reply = json_response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        banned_phrases = [
            "buy a house", "buy property", "invest in", "buy stocks", "buy shares",
            "purchase real estate", "real estate investment", "flip houses",
            "bitcoin", "crypto", "ethereum", "dogecoin", "nft", "forex trading",
            "options trading", "day trading", "penny stocks", "short selling",
            "invest in mcdonald", "invest in apple", "invest in tesla", "buy tesla", "buy amazon stock",
            "fake documents", "falsify income", "dispute all collections", "skip payments",
            "use someone else's credit", "open fake account", "hack", "scam", "fraud",
            "payday loan", "title loan", "borrow from loan shark", "use cash advance",
            "hide income", "evade taxes", "offshore account", "fake deduction"
        ]

        if any(phrase in reply.lower() for phrase in banned_phrases):
            reply = "⚠️ Sorry, I can't give specific investment or real estate advice. For personalized financial planning, please consult a licensed professional."

        if reply:
            conversation_history.append({"role": "assistant", "content": reply})
            return reply
        else:
            return "LLaMA responded but gave no message."

    except Exception as e:  
        return f"LLaMA error: {str(e)}"

def classify_user_input(question_text, user_input):
    # If the user input looks like a number, treat it as an answer
    if re.search(r'\d', user_input):
        return "answer"
    prompt = (
        f"You asked: '{question_text}'\n"
        f"The user replied: '{user_input}'\n"
        "Is this an Answer, a Question, or Neither? Respond with one word."
    )
    response = ask_llama(prompt)
    return response.lower().strip()

def get_human_question(field, default_prompt):
    """Ask LLaMA to reword this fixed question only."""
    prompt = (
        f"You are a helpful assistant trying to estimate a user's credit score.\n"
        f"You must ask about this field ONLY: '{field}'.\n"
        f"Here is the original form question: '{default_prompt}'.\n"
        f"Please rewrite it to sound more natural and conversational.\n"
        f"Ask ONE question only. Do NOT change the meaning."
    )
    return ask_llama(prompt)



def run_field_collection():
    print("\n🎯 Let's build your credit profile. Type 'skip' to skip or 'exit' to quit.")
    user_data = {}
    user_questions = []
    total_fields = len(field_questions)


    for i, (field, original_question) in enumerate(field_questions.items(), start=1):
        attempts = 0
        clarification = None


        while attempts < 3:
            print(f"\n📌 Question {i} of {total_fields}")
            human_question = get_human_question(field, original_question)
            print(f"LLaMA: {human_question}")
            user_input = input("You: ").strip()


            if user_input.lower() in ["exit", "quit"]:
                print("👋 Exiting early. Thanks for using FicoBuddy!")
                return user_data, user_questions


            if user_input.lower() in ["skip", "s"]:
                print("⏭️ Skipping this one.")
                break


            classification = classify_user_input(human_question, user_input)


            if classification == "answer":
                val = extract_clarified_number(user_input, field, clarification)


                if val is not None:
                    user_data[field] = val
                    print(f"✅ Got it: {val}")
                    break
                else:
                    # Try to clarify with LLaMA
                    clarification = ask_llama(
                        f"The user said: '{user_input}' in response to: '{human_question}'. "
                        f"Did they mean a number? Suggest a corrected version."
                    )
                    print(f"🤖 LLaMA: {clarification}")
            elif classification == "question":
                user_questions.append(user_input)
                print("📝 Saved your question — we'll answer it after this.")
            else:
                print("⚠️ That didn’t seem helpful. Try a number like 10%, $5000, or 3 times.")


            attempts += 1


    return user_data, user_questions

def test_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                print(" Ollama is running. Available models:")
                for model in models:
                    print(f"   - {model['name']}")
                return True
            else:
                print("⚠️  Ollama is running but no models are installed.")
                print("   Run: ollama pull llama3.2")
                return False
        return False
    except:
        print(" Cannot connect to Ollama. Make sure it's running:")
        print("   Run: ollama serve")
        return False

def extract_number(text, field_name=None):
    print(f"[extract_number] text='{text}', field_name='{field_name}'")
    text = text.lower().strip()

    # handle percentages
    if field_name == "Credit_Utilization_Ratio":
        percent_match = re.search(r'(\d+\.?\d*)\s*(%|percent)?', text)
        if percent_match:
            val = float(percent_match.group(1))
            return val / 100 if val > 1 else val

    # handle delayed payments / inquiries
    if field_name in ["Num_of_Delayed_Payment", "Num_Credit_Inquiries"]:
        if any(word in text for word in ["no", "none", "zero", "never", "0"]):
            return 0.0
        if any(word in text for word in ["once", "one", "1"]):
            return 1.0
        if any(word in text for word in ["twice", "two", "2"]):
            return 2.0
        if "few" in text:
            return 3.0

    # handle credit history age
    if field_name == "Credit_History_Age":
        age_match = re.search(r'(\d+\.?\d*)\s*(years|yrs|year)?', text)
        if age_match:
            return float(age_match.group(1))

    # handle $ amounts, "5k", "1.2 million", etc.
    text_clean = text.replace("$", "").replace(",", "").replace("usd", "")
    k_match = re.search(r'(\d+\.?\d*)\s*(k|thousand)', text_clean)
    if k_match:
        return float(k_match.group(1)) * 1000
    m_match = re.search(r'(\d+\.?\d*)\s*(m|million)', text_clean)
    if m_match:
        return float(m_match.group(1)) * 1_000_000
    number_match = re.findall(r'\d+\.?\d*', text_clean)
    if number_match:
        return float(number_match[0])

    return None

def extract_clarified_number(original_input, field_name, clarification_reply):
    """If clarification contains a corrected value (e.g. 25 years), extract that instead of 'yes'."""
    if clarification_reply and original_input.lower() in ["yes", "yeah", "yep", "correct"]:
        return extract_number(clarification_reply, field_name=field_name)
    return extract_number(original_input, field_name=field_name)


from collections import OrderedDict


field_questions = OrderedDict({
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
})

ordered_fields = ["name"] + list(field_questions.keys())

chat_state = {
    "current_index": 0,
    "user_data": {},
    "user_questions": [],
    "clarification": None
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

    tips_prompt = (
    f"The user has a credit score of {int(prediction)} ({category}). "
    f"Their credit utilization is {user_data.get('Credit_Utilization_Ratio', 0)*100:.0f}%, "
    f"they've had {user_data.get('Num_of_Delayed_Payment', 0)} late payments, "
    f"and owe ${user_data.get('Outstanding_Debt', 0):,.0f}. "
    f"Give 3 specific, actionable tips to improve their credit score. Be friendly and encouraging."
)
    print(ask_llama(tips_prompt))

    more = input("\n🎯 Would you like more advanced financial tips? (yes/no): ")


    if more in ["yes", "y"]:
        conversation_history.append({
            "role": "system",
            "content": (
                "You are a cautious, helpful financial advisor. "
                "Give general, non-specific financial strategies to improve credit and money habits. "
                "Never suggest buying specific stocks, real estate, or name-brand investments. "
                "Avoid regulated financial advice or anything that could be construed as investment advice. "
                "Focus on safe, educational, actionable guidance."
            )
        })

    print("\nAdvanced Strategies:")
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

    print("\n💬 You can now ask me anything about your credit score, finances, or how to improve.")
    print("Type 'exit' to finish the session.\n")

    while True:
        follow_up = input("You: ").strip()
        if follow_up.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Thanks for using FicoBuddy!")
            return
        break
    reply = ask_llama(follow_up)
    print(f"LLaMA: {reply}\n")


        
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
    print("\n🎯 Welcome to the FicoBuddy Credit Score Assistant!")
    print("=" * 50)


    user_name = input("👋 Before we begin, what's your name? ").strip().title()
    print(f"\nNice to meet you, {user_name}! Let's get started.")


    conversation_history.insert(1, {
        "role": "user",
        "content": f"My name is {user_name}."
    })

    user_data, user_questions = run_field_collection()


    # Run the model
    if len(user_data) >= 5:
        run_credit_optimizer(user_data)
    else:
        print("\n⚠️ Not enough valid answers to estimate your credit score.")


    # Answer follow-up user questions
    if user_questions:
        print("\n You asked some good questions. Let’s review them now:\n")
        for q in user_questions:
            print(f"❓ {q}")
            print("💬", ask_llama(q))

    if len(user_data) >= 5:
        run_credit_optimizer(user_data)
    else:
        print("\n⚠️  Not enough responses to make an accurate prediction.")
        print("Please answer at least 5 questions for a credit score estimate.")

def start_ngrok():
    print("Starting ngrok tunnel...")
    ngrok_process = subprocess.Popen(["ngrok", "http", "--domain=known-highly-treefrog.ngrok-free.app", "5000"], stdout=subprocess.DEVNULL)
    time.sleep(5)  

    try:
        ngrok_api = requests.get("http://localhost:4040/api/tunnels")
        public_url = ngrok_api.json()["tunnels"][0]["public_url"]
        print(f"\n Your Flask API is publicly available at: {public_url}/api/chat\n")
    except Exception as e:
        print(f"Failed to fetch ngrok URL: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        from flask import Flask, request, jsonify
        from flask_cors import CORS

        app = Flask(__name__)
        CORS(app, origins=[
            "http://localhost:59824",
            "http://localhost:4200"
        ])

        @app.route("/api/chat", methods=["GET", "POST"])
        def chat_endpoint():
            try:
                if request.method == "GET":
                    return ('', 204)  # No Content — silent success

                data = request.get_json()
                user_message = data.get("message", "").strip()


                if not user_message:
                    return jsonify({"error": "Missing 'message'"}), 400


                if user_message.lower() in ["__start__", "hello", "hi", "hey", "start"]:
                    chat_state["current_index"] = 0
                    chat_state["user_data"] = {}
                    chat_state["user_questions"] = []
                    chat_state["clarification"] = None

                    return jsonify({
                        "response": "👋 Hello! I'm FicoBuddy, your credit assistant.\n\nWhat's your first name?",
                        "showRecommendations": False
                    })


                i = chat_state["current_index"]
                if i >= len(ordered_fields):
                    score, category, tips = process_user_data(chat_state["user_data"])
                    return jsonify({
                        "response": f"📊 Your estimated credit score is **{score}** ({category}).\n\nTips: {tips}",
                        "showRecommendations": True,
                        "continueChat": True
                    })

                field = ordered_fields[i]
                if field == "name":
                    if len(user_message) < 2 or not user_message.replace(" ", "").isalpha():
                        return jsonify({
                            "response": "👋 Please enter just your first name (letters only).",
                            "showRecommendations": False
                        })

                    chat_state["user_data"]["name"] = user_message.title()
                    chat_state["current_index"] += 1
                    if chat_state["current_index"] < len(ordered_fields):
                        next_field = ordered_fields[chat_state["current_index"]]
                        next_q = get_human_question(next_field, field_questions[next_field])
                        return jsonify({
                            "response": f"Nice to meet you, {user_message.title()}! {next_q}",
                            "showRecommendations": False
                        })
                    else:
                        # Immediately calculate and show the score and tips in chat
                        score, category, tips = process_user_data(chat_state["user_data"])
                        return jsonify({
                            "response": f"📊 Your estimated credit score is **{score}** ({category}).\n\nTips: {tips}"
                        })

                original_q = field_questions[field]
                classification = classify_user_input(original_q, user_message)
                print(f"[classify_user_input] '{user_message}' classified as '{classification}'")

                if classification == "answer":
                    val = extract_clarified_number(user_message, field, chat_state["clarification"])
                    if val is not None:
                        chat_state["user_data"][field] = val
                        chat_state["current_index"] += 1
                        chat_state["clarification"] = None

                        if chat_state["current_index"] < len(ordered_fields):
                            next_field = ordered_fields[chat_state["current_index"]]
                            next_q = get_human_question(next_field, field_questions[next_field])
                            return jsonify({
                                "response": next_q,
                                "showRecommendations": False
                            })
                        else:
                            # Immediately calculate and show the score and tips in chat
                            score, category, tips = process_user_data(chat_state["user_data"])
                            return jsonify({
                                "response": f"📊 Your estimated credit score is **{score}** ({category}).\n\nTips: {tips}"
                            })

                    else:
                        chat_state["clarification"] = ask_llama(
                            f"The user said: '{user_message}' in response to: '{original_q}'. "
                            f"Did they mean a number? Suggest a corrected version."
                        )
                        return jsonify({
                            "response": f"🤖 Just to clarify — did you mean: {chat_state['clarification']}?",
                            "showRecommendations": False
                        })


                elif classification == "question":
                    chat_state["user_questions"].append(user_message)
                    return jsonify({
                        "response": "📝 Got your question! I'll answer it after we finish your profile.",
                        "showRecommendations": False
                    })


                else:
                    return jsonify({
                        "response": "⚠️ Hmm, that didn't look like a number. Try something like 10%, $5000, or 3 times.",
                        "showRecommendations": False
                    })


                if chat_state["current_index"] < len(ordered_fields):
                    next_field = ordered_fields[chat_state["current_index"]]
                    next_q = get_human_question(next_field, field_questions[next_field])
                    return jsonify({
                        "response": next_q,
                        "showRecommendations": False
                    })
                else:
                    # Immediately calculate and show the score and tips in chat
                    score, category, tips = process_user_data(chat_state["user_data"])
                    return jsonify({
                        "response": f"📊 Your estimated credit score is **{score}** ({category}).\n\nTips: {tips}"
                    })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.route("/api/score", methods=["POST"])
        def score_endpoint():
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "Missing JSON body"}), 400

                score, category, tips = process_user_data(data)

                return jsonify({
                    "score": int(score),
                    "category": category,
                    "tips": tips,
                    "showRecommendations": True
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        start_ngrok()
        app.run(port=5000)
    else:
        main()
