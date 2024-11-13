import os
import pickle
import difflib
import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Ensure nltk resources are downloaded
nltk.download('wordnet')

# Define the dictionary path and user data folder
DICTIONARY_PATH = "dictionary.py"
USER_DATA_FOLDER = "user_data"

# Create user data folder if it doesn't exist
if not os.path.exists(USER_DATA_FOLDER):
    os.makedirs(USER_DATA_FOLDER)

# Load the dictionary from the file
def load_dictionary():
    try:
        import dictionary
        return dictionary.response_dict
    except ImportError:
        return {}

def save_dictionary(dictionary_data):
    with open(DICTIONARY_PATH, 'w') as f:
        f.write(f"response_dict = {dictionary_data}")

# Load or create user-specific data file
def load_user_data(username):
    user_file = os.path.join(USER_DATA_FOLDER, f"{username}.pkl")
    if os.path.exists(user_file):
        with open(user_file, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def save_user_data(username, data):
    user_file = os.path.join(USER_DATA_FOLDER, f"{username}.pkl")
    with open(user_file, 'wb') as f:
        pickle.dump(data, f)

# Initialize or load the model
def load_or_train_model():
    model_file = 'chatbot_model.pkl'
    vectorizer_file = 'vectorizer.pkl'
    dictionary_data = load_dictionary()

    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    else:
        if dictionary_data:
            vectorizer = TfidfVectorizer()
            responses = list(dictionary_data.values())
            keys = list(dictionary_data.keys())
            X = vectorizer.fit_transform(keys)
            with open(model_file, 'wb') as f:
                pickle.dump(responses, f)
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(vectorizer, f)
            return responses, vectorizer
        else:
            raise ValueError("Dictionary is empty or missing.")

# Synonym handling
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
    return ' '.join(expanded)

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
        "couldn't": "could not"
    }
    for contraction, expanded in contractions.items():
        query = query.replace(contraction, expanded)
    return query

def remove_punctuation(query):
    return re.sub(r'[^\w\s]', '', query)

def get_best_match(query, keys, vectorizer):
    query = expand_contractions(query)
    query = expand_query_with_synonyms(query)
    query = remove_punctuation(query)
    query_vector = vectorizer.transform([query.lower()])
    key_vectors = vectorizer.transform(keys)
    similarities = cosine_similarity(query_vector, key_vectors)
    most_similar_idx = similarities.argmax()
    return keys[most_similar_idx]

def chatbot_response(user_input, username):
    # Special case for commands to clear data
    if "clear your memory" in user_input.lower() or "clear your data" in user_input.lower():
        return clear_user_data(username)

    # Special case for deleting all data
    if "delete my data" in user_input.lower():
        return delete_all_user_data(username)

    # Load the dictionary and user-specific data
    dictionary_data = load_dictionary()
    user_data = load_user_data(username)
    responses, vectorizer = load_or_train_model()

    user_input = user_input.lower()  # Normalize input

    # Normalize user input (for matching variations)
    normalized_input = expand_contractions(user_input)
    normalized_input = remove_punctuation(normalized_input)  # Remove punctuation
    normalized_input = expand_query_with_synonyms(normalized_input)  # Expand with synonyms

    # Check for exact match in the predefined dictionary first
    if normalized_input in dictionary_data:
        return dictionary_data[normalized_input]

    # Check for exact match in user-specific data
    if normalized_input in user_data:
        return user_data[normalized_input]

    # Check for partial match using similarity and synonyms
    keys = list(dictionary_data.keys())
    best_match = get_best_match(normalized_input, keys, vectorizer)
    if difflib.SequenceMatcher(None, normalized_input, best_match).ratio() > 0.7:
        return dictionary_data[best_match]

    # If no match is found, prompt the user to teach the bot
    return learn_new_information(user_input, username)

def learn_new_information(question, username):
    response = input(f"I don't know the answer to '{question}'. Please provide an answer: ")
    normalized_question = expand_contractions(question.lower())
    expanded_question = expand_query_with_synonyms(normalized_question)

    user_data = load_user_data(username)
    dictionary_data = load_dictionary()

    user_data[expanded_question] = response
    save_user_data(username, user_data)

    dictionary_data[expanded_question] = response
    save_dictionary(dictionary_data)

    return "Thanks for teaching me! I will remember this for next time."

def delete_all_user_data(username):
    user_file = os.path.join(USER_DATA_FOLDER, f"{username}.pkl")
    
    if os.path.exists(user_file):
        os.remove(user_file)
        print("Bot: All your saved data, including your name and responses, has been deleted.")
        return True
    else:
        print("Bot: No data found to delete.")
        return False

if __name__ == "__main__":
    print("Welcome to the chatbot!")
    
    while True:
        username = input("Please enter your name: ").strip().lower()
        user_data = load_user_data(username)

        # If no data found, create a new entry
        if not user_data:
            print(f"Created a new profile for '{username}'")
            save_user_data(username, {})  # Create an empty profile for the new user
            break
        else:
            break

    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye! Have a great day!")
            break

        # Delete user data if requested
        if "delete my data" in user_input.lower():
            if delete_all_user_data(username):
                print("Please enter your name again to create a new profile.")
                username = input("Please enter your name: ").strip().lower()
                user_data = load_user_data(username)
                continue
        
        # Get chatbot response
        print("Bot:", chatbot_response(user_input, username))