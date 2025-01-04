import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("※ CHATBOT ※")

    menu = ["Home", "Chat History", " About the Chatbot "]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Dive into the world of smart conversations! Type a message, press Enter, and let's create something extraordinary together.")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye', 'exit', 'quit', 'good bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Chat History":
        st.header("Chat History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == " About the Chatbot ":
        st.write("""The goal of this project is to serve as an intelligent conversational interface that can understand and respond to user queries effectively. Its primary goal is to identify the user’s intent behind a message, such as asking a question,making a request,
                  or seeking guidance, and then provide an appropriate response or take a corresponding action.By leveraging natural language processing techniques, the chatbot interprets the user’s input, extracts key information,
                  and matches it to predefined intents and entities. This allows it to automate tasks, answer queries, or provide recommendations without the need for human intervention.""")

        st.subheader("Project Overview:")

        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")

        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "help", "name", "hobby")
        - Entities: The entities extracted from user input (e.g. "How can i help you", "Do you have a name?", "What is your hobby?")
        - Text: The user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")

        st.write("""These chatbots can serve a wide range of applications, from customer support and virtual assistants to educational platforms and healthcare. With continuous training on diverse datasets, intent-based chatbots can evolve to handle complex queries, improve response accuracy, and provide a seamless user experience.
                  Moreover, advancements in AI and NLP ensure that such chatbots will continue to grow in capabilities, bridging the gap between human and machine interaction. However, ethical considerations, including user privacy and unbiased responses, must remain central to their development. In conclusion, intent-based chatbots 
                  represent a significant step forward in leveraging AI for meaningful and intelligent communication. """)
        
    st.markdown("---")
    st.image("a.png", use_container_width=True)
    st.markdown("""
        <div style='text-align: center'>
            <p>Made with ❤️ using Python and Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()