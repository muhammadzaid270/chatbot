import tkinter as tk
from tkinter import simpledialog, scrolledtext
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import time

# Ensure nltk resources are downloaded
nltk.download("wordnet")
from nltk.corpus import wordnet as wn

# Load pre-defined dictionary
def load_dictionary():
    try:
        import dictionary
        return dictionary.response_dict
    except ImportError:
        return {}

# Expand contractions
def expand_contractions(query):
    contractions = {
        "who's": "who is",
        "what's": "what is",
        "it's": "it is",
        "he's": "he is",
        "she's": "she is",
        "that's": "that is",
        "they're": "they are",
        "I'm": "I am",
        "you're": "you are",
        "we're": "we are",
        "haven't": "have not",
        "hasn't": "has not",
        "don't": "do not",
        "doesn't": "does not",
        "can't": "cannot",
        "couldn't": "could not",
    }
    for contraction, expanded in contractions.items():
        query = query.replace(contraction, expanded)
    return query

# Remove punctuation
def remove_punctuation(query):
    return re.sub(r"[^\w\s]", "", query)

# Expand query with synonyms
def find_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def expand_query_with_synonyms(query):
    words = query.split()
    expanded = set(words)
    for word in words:
        expanded.update(find_synonyms(word))
    return " ".join(expanded)

# Temporary memory for learned responses
temporary_memory = {}

# Pre-defined patterns for more natural queries
patterns = {
    r"who\s+is\s+\w+": "Could you specify the name you're asking about? For example: 'Who is Messi?'",
    r"who\s+are\s+you": "I'm a chatbot, here to help!",
    r"\b(what are you)\b": "I’m a chatbot, here to help!",
    r"\b(how are you)\b": "I'm doing great, thank you for asking!",
}

# Function to match patterns
def match_pattern(query, patterns):
    for pattern, response in patterns.items():
        if re.search(pattern, query):
            return response
    return None

# Chatbot response function
def chatbot_response(user_input):
    dictionary_data = load_dictionary()

    # Train vectorizer
    combined_keys = list(dictionary_data.keys()) + list(temporary_memory.keys())
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(combined_keys)

    # Normalize input
    normalized_input = expand_contractions(user_input.lower())
    normalized_input = remove_punctuation(normalized_input)

    # 1. **Check for exact match in dictionary**
    if normalized_input in dictionary_data:
        return dictionary_data[normalized_input]

    # 2. Check for exact match in temporary memory
    if normalized_input in temporary_memory:
        return temporary_memory[normalized_input]

    # 3. Match input patterns for common phrases
    pattern_response = match_pattern(normalized_input, patterns)
    if pattern_response:
        return pattern_response

    # 4. Check for best match across combined keys
    if combined_keys:
        best_match, similarity = get_best_match(normalized_input, combined_keys, vectorizer)
        if similarity > 0.7:
            if best_match in dictionary_data:
                return dictionary_data[best_match]
            elif best_match in temporary_memory:
                return temporary_memory[best_match]

    # 5. Learn new information if no match found
    return learn_new_information(user_input)

# Function to find the best match using cosine similarity
def get_best_match(query, keys, vectorizer):
    query_vector = vectorizer.transform([query.lower()])
    key_vectors = vectorizer.transform(keys)
    similarities = cosine_similarity(query_vector, key_vectors)
    most_similar_idx = similarities.argmax()
    best_match = keys[most_similar_idx]
    return best_match, similarities[0][most_similar_idx]

# Learn new information temporarily using a GUI prompt
def learn_new_information(question):
    # Create a pop-up window to get the user's input
    response = simpledialog.askstring("New Information", f"I don't know the answer to '{question}'. Please provide an answer:")
    if response:
        temporary_memory[question] = response
        return "Thanks for teaching me! I will remember this until the program ends."
    else:
        return "No answer provided. I will not remember this."

# Create the main window for the chatbot GUI
def create_gui():
    window = tk.Tk()
    window.title("Chatbot")
    window.geometry("800x600")  # Updated window size for a better fit

    # Add a title label
    title_label = tk.Label(window, text="Chatbot", font=("Arial", 30), bg="lightblue", width=30)
    title_label.pack(pady=10)

    # Add a frame to hold the conversation
    conversation_frame = tk.Frame(window)
    conversation_frame.pack(padx=10, pady=10)

    # Add a chat area (scrollable text widget for chat)
    chat_area = scrolledtext.ScrolledText(conversation_frame, height=15, width=100, font=("Arial", 14), wrap=tk.WORD, bd=0, bg="#f0f0f0")
    chat_area.pack(padx=10, pady=10)

    # Add an input area for the user
    user_input_area = tk.Entry(window, font=("Arial", 14), width=70)
    user_input_area.pack(padx=10, pady=10)

    # Function to handle user input and chatbot response with typewriter effect
    def typewriter_effect(message, speed=5):
        for char in message:
            chat_area.insert(tk.END, char)
            chat_area.yview(tk.END)
            chat_area.update()
            time.sleep(speed / 1000)
        chat_area.insert(tk.END, "\n")
        chat_area.yview(tk.END)

    # Function to handle the sending of messages
    def send_message(event=None):
        user_input = user_input_area.get()
        if user_input.strip():
            # Display user's input in the chat area
            chat_area.insert(tk.END, f"You: {user_input}\n")
            chat_area.yview(tk.END)

            # Get bot's response and display it with the typewriter effect
            bot_response = chatbot_response(user_input)
            typewriter_effect(bot_response)

            # Clear the input area
            user_input_area.delete(0, tk.END)

    # Bind the Enter key to send messages
    user_input_area.bind("<Return>", send_message)

    # Add a button to send the message
    send_button = tk.Button(window, text="Send", font=("Arial", 14), command=send_message, bg="lightgreen")
    send_button.pack(pady=10)

    # Start the GUI loop
    window.mainloop()

if __name__ == "__main__":
    create_gui()