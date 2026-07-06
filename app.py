import os
import re
import spacy
import joblib  
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# base directory and model import
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "Compliant_NB.joblib")
vect_path = os.path.join(BASE_DIR, "models", "vectorizer_for_compliant_NB.pkl")
# ---------------------------------------------------------------------------------------------------------------------
# model loading
model = joblib.load(model_path)
vect=joblib.load(vect_path)
# ---------------------------------------------------------------------------------------------------------------------
# transformers
emotion_classifier = pipeline("text-classification",model="j-hartmann/emotion-english-distilroberta-base",top_k=None)
polarity_classifier = pipeline("sentiment-analysis",model="cardiffnlp/twitter-roberta-base-sentiment-latest")
# ----------------------------------------------------------------------------------------------------------------------

# text preprocessing
# these are the text preprocessing techniques used in this application

# 1.regular expression and normalization 
def regex_normalize(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower().replace('\n', ' ').replace('\t', ' ') # Lowercase texts and replace /n & /t in the text
    
    text = re.sub(r'\b[0-9/]*X{2,}[0-9/]*\b', '', text) # removes XX/XX/XX19 ,XX/XX/XXXX, XXXX/XXXX/XXXX

    text = re.sub(r'\bx+\b', '', text, flags=re.IGNORECASE) #  replaces xxxx blocks out completely

    text = re.sub(r'\{?\$[0-9.,]+\}?|\{?[0-9.,]+\$\}?', '[MONEY]', text) # replaces any amoount money with currency symbol into token( [MONEY] )
    
    text = re.sub(r'\.{3,}', '', text) # replaces triple dots with nothing in its place

    text = re.sub(r'\d+', '', text) # replaces the numbers like year, two-digit numbers, etc

    text = re.sub(r'[^a-zA-Z\s]', ' ', text) # replaces -(hyphen) before words
    
    text = re.sub(r"\'", "", text) # Removes any backslash that immediately precedes a single quote

    text = re.sub(r'%\s*[0-9.,]+|[0-9.,]+\s*%', '[PERCENTAGE]', text) # replaces the numbers with % with improper placements into token( [PERCENTAGE] )
    
    text = re.sub(r'\s+', ' ', text).strip() # Collapse multiple spaces into a single space

    text = text.replace(r"\'\'", '"').replace(r"\'", "'") # replaces \'' and other slash symbols with single quotaion
    
    return text

# 2.Lemmatization
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def lemmatize_row(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    doc = nlp(text)
    
    # Extract the lemma only if it is NOT punctuation, a symbol, or a standalone period
    lemmas = [token.lemma_ for token in doc if not token.is_punct and token.pos_ != "SYM" and token.lemma_ != "."]
    
    return " ".join(lemmas)

# 3.stopwords removal
from spacy.lang.en.stop_words import STOP_WORDS

stopwords = set(STOP_WORDS)

custom_stopwords = {"complaint", "case", "number", "regards", "listed", 
                    "attached", "document", "pdf", "company", "bank", 
                    "loan", "citizen", "account","receive", "say", 
                    "tell", "call", "state", "go", "would", "could",
                    "citibank", "citcard", "recieve", "fradulent", 
                    "website", "show", "tell", "say", "talk", "ask",
                    "tell", "say","talk", "advise", "receive", "state", 
                    "include", "question", "website", "show", "transferee",
                    "husband", "father", "law", "attorney", "rep", "supervisor", 
                    "name","numbersince", "forth", "till", "today", 
                    "yesterday", "tomorrow","sentence", "contact", "supply", 
                    "request", "letter", "email", "writing"
                   }

final_stopwords = stopwords.union(custom_stopwords) # Combine them into a single high-speed lookup set

def filter_tokens(text): # FILTER FUNC
    if not isinstance(text, str) or not text.strip():
        return ""
    
    words = text.split()
 
    filtered_words = [word for word in words if word not in final_stopwords and len(word) > 2]
    
    return " ".join(filtered_words)

# vectorization
# this is the vectorization parameters used in this application, but the actual vectoraization is imported abov
#vectorizer = TfidfVectorizer(max_features=2500,norm='l2',min_df=2,ngram_range=(1, 2))

# end of preprocessing block
# ---------------------------------------------------------------------------------------------------------------------

# webpage name and icon
st.set_page_config(
    page_title="Compliant Prioritizer",
    page_icon="🎫",
    layout="centered"
)
# ----------------------------------------------------------------------------------------------------------------------

# Sidebar
# About title
st.sidebar.title("About")

# sidebar markdown 1
st.sidebar.markdown("""
### 🎯 WHAT IT DOES:
This is a Machine Learning application recognizes seven categories of the Banking Complaints of the 
customers using Natural Language Processing ,ML model and Transformers, it recognizes the problem faced by the consumer
by looking at the compliant and categorize it and it finds the emotion and severity of the problem
based on the context of the compliant. By evaluating the category, emotions and severity of the problem 
in the consumer's complaint we can prioritize the complaint ticket and resolve the problem for the consumers, which shows the 
reliablity of the services provided and helps in customer retention for the banks.
""")

# sidebar markdown 2
st.sidebar.markdown("""
#### 🫠 WHAT IT IS TRAINED ON:
This model specifically trained on Seven categories of complaint tickets like Banking Services, Credit Card, Consumer Loan, 
Credit Reporting, Debt collection, Mortgage and Student Loan 

""")
st.sidebar.subheader("🛠️ Tech Stack")
tech_stack = [
    "Python",
    "Pandas",
    "Scikit-Learn",
    "Spacy",
    "Transformers",
    "Streamlit",
]
for tech in tech_stack:
    st.sidebar.markdown(f"- **{tech}**")

# github link bar
st.sidebar.subheader("🔗 Source Code")
st.sidebar.link_button("💻 Go to GitHub Repository", "https://github.com/cecsranjethaswinr23-collab/Complaint_Prioritizer_NLP", use_container_width=True)

# author bar
st.sidebar.markdown("### 👨‍💻 Developed By")
st.sidebar.markdown("**Ranjeth Aswin Ravindran**")
st.sidebar.caption("Data Scientist")

# Email bar
st.sidebar.markdown("""
📧 **Contact:** cecsranjethaswinr23@gmail.com
""")
# end of about section
# ------------------------------------------------------------------------------------------------------------------------------------

# Main page 

# 1. title
st.title("Complaint Prioritizer 🎫")
st.markdown(""" This application recognizes the severity of your problem and categorize it. The complaints with high risk for the consumers
will be recognized by the application and prioritize the tickets for immediate resolving of the problem for better consumer services...
""")

st.write("---")
# -------------------------------------------------------------------------------------------------------------------


# 2.complaint box and hit button
def complaint_input_section():
    # Text area with a disappearing watermark (placeholder)
    user_complaint = st.text_area(
        label="Enter your banking complaint:",
        placeholder="e.g., The refunds still not credited but the application shows the refund has been made."
    )
    
    if st.button("Submit Ticket 👇"):
        if user_complaint.strip() == "":
            st.warning("Please type your complaint before submitting.")
        else:
            with st.spinner("Processing ticket variables..."):
                results = analyze_complaint(user_complaint)
                return results
    return None
# complaint box end
# ---------------------------------------------------------------------------------------------------------------------

# 3.prediction
def predict_complaint(text):
    the_classes={'Bank account or service': 0,'Consumer Loan': 1,
                 'Credit card': 2,'Credit reporting': 3,
                 'Debt collection': 4,'Mortgage': 5,'Student loan': 6}

    text = regex_normalize(text)             # Remove patterns using regular expression
    text = lemmatize_row(text)               # Lemmatization
    text = filter_tokens(text)               # Remove stopwords 
    text_vector = vect.transform([text])     # Convert to TF-IDF features
    prediction = model.predict(text_vector)  # Predicts the class 

    complaint = [k for k, v in the_classes.items() if v == int(prediction[0])]
    return complaint[0]


def analyze_complaint(complaint_text):

    category = predict_complaint(complaint_text)   # predicts the complaint ticket category
    emotion_output = emotion_classifier(complaint_text)[0]   # Emotion model
    top_emotion = max(emotion_output, key=lambda x: x["score"]) # gets the maximum probability of emotion
    polarity = polarity_classifier(complaint_text)[0]     # text severity model checks for the negativity in the text
    sentiment_label = polarity["label"].lower()   # 'negative', 'neutral', or 'positive'
    sentiment_score = polarity["score"]          # The confidence float (e.g., 0.945)

   
    # Define high-stakes banking categories that require rapid legal/financial intervention
    critical_categories = ["Credit card", "Debt collection", 
                           "Mortgage", "Bank account or service"]
    
    # High-stress emotional flags
    volatile_emotions = ["anger", "fear", "disgust"]

    # CRITICAL EMERGENCY
    # for fearing customers(like losing something or  made a mistake)
    if sentiment_label == "negative" and top_emotion["label"] == 'fear' and sentiment_score >= 0.90 and category in critical_categories:
        priority_message = "🚨 CRITICAL: Your complaint involves a high-risk scenario. This complaint will be  handled with the highest level of urgency and has been put at the top of our priority list for immediate resolution. the problem is passed to the team and they will be actively working on a solution for the problem."
    
    elif sentiment_label == "negative" and top_emotion["label"] == 'fear' and (sentiment_score <= 0.89 and sentiment_score >=70) and category in critical_categories:
        priority_message = "⚠️ HIGH PRIORITY: High distress detected. Your ticket has been escalated ahead of standard inquiries. We treat this issue as a top priority for our team. The error is already escalated to our team for immediate correction."
    
    # MEDIUM PRIORITY
    # angry consumers
    elif top_emotion["label"]== 'anger' and sentiment_label == "negative" and sentiment_score >= 0.75 and category in critical_categories:
        priority_message = "📢 Your ticket has been escalated ahead and inquired thoroughly and resolved as soon as possible. We deeply regret the severe frustration this has caused you. We are fully committed to resolving this completely so it is made right. "
    
    elif top_emotion["label"]== 'disgust' and sentiment_label == "negative" and sentiment_score >= 0.75 and category in critical_categories:
        priority_message = "📢 Your ticket has been escalated ahead and inquired thoroughly and resolved as soon as possible. This is completely unacceptable and does not reflect our standards. We will be taking immediate action to isolate and fix this issue."

    # less emotional and fixable problems
    else:
        priority_message = "✅ STANDARD QUEUE: Please accept our sincere apologies for the error. We are correcting the unfortunate errors faced by our customers. Soon the problem will be resolved. Thank you for informing us."

    # =====================================================================

    return {
        "category": category,
        "emotion": top_emotion["label"].upper(),
        "emotion_score": top_emotion["score"],
        "sentiment": polarity["label"].upper(),
        "severity_score": sentiment_score,
        "priority_status": priority_message  # The custom priority decision string
    }
analysis_results = complaint_input_section()

# If a ticket was successfully submitted and evaluated, display it!
if analysis_results is not None:
    st.success("Analysis Complete!")
    
    # Display the results neatly on the screen
    st.write(f"**Predicted Category:** {analysis_results['category']}")
    st.write(f"**Detected Emotion:** {analysis_results['emotion']} ({analysis_results['emotion_score']:.2%})")
    st.write(f"**TEXT SEVERITY:** {analysis_results['sentiment']} ({analysis_results['severity_score']:.2%})")
    
    st.info(analysis_results["priority_status"])