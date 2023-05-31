import streamlit as st
from transformers import pipeline
import spacy
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string

# Initialize NLP pipelines
sentiment_analysis = pipeline("sentiment-analysis")
summarization = pipeline("summarization")
ner = pipeline("ner")
translation = pipeline("translation_en_to_fr")
nlp = spacy.load("en_core_web_sm")

# Preprocess text for topic modeling
def preprocess_text(text):
    return preprocess_string(text)

# Train a topic model (e.g., LDA) using Gensim
def train_topic_model(preprocessed_text):
    dictionary = corpora.Dictionary([preprocessed_text])
    corpus = [dictionary.doc2bow(text) for text in [preprocessed_text]]
    lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)
    return dictionary, corpus, lda_model

# Streamlit app
def main():
    st.title("NLP Dashboard")

    # Input text
    user_input = st.text_area("Enter your text here:")

    # Choose NLP task
    task = st.selectbox(
        "Choose an NLP task:",
        [
            "Sentiment Analysis",
            "Text Summarization",
            "Named Entity Recognition",
            "Part-of-Speech Tagging",
            "Topic Modeling",
            "Machine Translation",
        ],
    )

    # Perform the selected task
    if st.button("Perform Task"):
        if task == "Sentiment Analysis":
            result = sentiment_analysis(user_input)
            st.write("Sentiment:", result[0]["label"])
            st.write("Confidence:", result[0]["score"])

        elif task == "Text Summarization":
            result = summarization(user_input)
            st.write("Summary:", result[0]["summary_text"])

        elif task == "Named Entity Recognition":
            result = ner(user_input)
            for entity in result:
                st.write(f"{entity['entity']}: {entity['word']}")

        elif task == "Part-of-Speech Tagging":
            doc = nlp(user_input)
            for token in doc:
                st.write(f"{token.text}: {token.pos_}")

        elif task == "Topic Modeling":
            # Preprocess the textpreprocessed_text = preprocess_text(user_input)

            # Train a topic model (e.g., LDA) using Gensim
            dictionary, corpus, lda_model = train_topic_model(preprocessed_text)

            # Display the topics
            for idx, topic in lda_model.print_topics(-1):
                st.write(f"Topic {idx}: {topic}")

        elif task == "Machine Translation":
            result = translation(user_input)
            st.write("Translation:", result[0]["translation_text"])

if __name__ == "__main__":
    main()
