import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load the pre-trained NLP model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Define a function to encode user and chatbot messages
def encode_messages(user_message, chatbot_message):
    # Convert messages to tensors
    user_message = tf.constant(user_message)
    chatbot_message = tf.constant(chatbot_message)

    # Encode messages using the NLP model
    encoded_user_message = model(user_message)
    encoded_chatbot_message = model(chatbot_message)

    return encoded_user_message, encoded_chatbot_message

# Define a function to calculate cosine similarity between user and chatbot messages
def calculate_cosine_similarity(encoded_user_message, encoded_chatbot_message):
    # Calculate cosine similarity
    cosine_similarity = tf.reduce_sum(encoded_user_message * encoded_chatbot_message) / (
        tf.linalg.norm(encoded_user_message) * tf.linalg.norm(encoded_chatbot_message)
    )

    return cosine_similarity

# Define a function to generate a chatbot response
def generate_chatbot_response(user_message):
    # Encode the user message
    encoded_user_message = model([user_message])

    # Initialize an empty list to store cosine similarities
    cosine_similarities = []

    # Iterate through the chatbot's training data
    for chatbot_message in chatbot_training_data:
        # Encode the chatbot message
        encoded_chatbot_message = model([chatbot_message])

        # Calculate cosine similarity between the user message and the chatbot message
        cosine_similarity = calculate_cosine_similarity(encoded_user_message, encoded_chatbot_message)

        # Append the cosine similarity to the list
        cosine_similarities.append(cosine_similarity)

    # Find the index of the chatbot message with the highest cosine similarity
    highest_cosine_similarity_index = np.argmax(cosine_similarities)

    # Return the chatbot's response
    return chatbot_training_data[highest_cosine_similarity_index]

# Load the chatbot's training data
chatbot_training_data = [
    "Hello",
    "How are you?",
    "I'm doing well, thank you. What can I do for you?",
    "My name is Bard.",
    "Goodbye",
]

# Start the conversation
while True:
    # Get the user's message
    user_message = input("Tilan: ")

    # Generate a chatbot response
    chatbot_response = generate_chatbot_response(user_message)

    # Print the chatbot's response
    print("My bot: " + chatbot_response)

    # Check if the user wants to end the conversation
    if user_message == "Goodbye":
        break
