import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import time
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

# Learn new information temporarily (using a Streamlit input box)
def learn_new_information(question):
    response = st.text_input(f"I don't know the answer to '{question}'. Please provide an answer:", key="learn_input")
    if response:
        temporary_memory[question] = response
        return "Thanks for teaching me! I will remember this until the program ends."
    else:
        return "No answer provided. I will not remember this."

# Create the main Streamlit interface
def create_streamlit_interface():
    st.title("Chatbot")

    # Store the conversation in a list
    conversation = []

    # Get user input with a unique key
    user_input = st.text_input("You:", key="user_input_1")

    if user_input:
        # Append user's input to conversation
        conversation.append(f"You: {user_input}")

        # Get the bot's response
        bot_response = chatbot_response(user_input)
        conversation.append(f"Bot: {bot_response}")

        # Display conversation
        for message in conversation:
            st.write(message)

        # Clear the input field after submission
        st.text_input("You:", value="", key="user_input_2")

if __name__ == "__main__":
    create_streamlit_interface()