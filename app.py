%%writefile app.py
import streamlit as st
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Load the dataset
dataset_path = '/content/hormozi_tweets.jsonl'
df = pd.read_json(dataset_path, lines=True)

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to tokenize a tweet
def tokenize_tweet(tweet):
    return tokenizer.encode(tweet, return_tensors='pt').to(device)

# Function to generate a new tweet
def generate_tweet(prompt, max_length=50):
    model.eval()
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to compute a reward based on feedback
def compute_reward(feedback):
    relevance_weight = 0.4
    clarity_weight = 0.2
    originality_weight = 0.2
    engagement_weight = 0.2
    reward = (feedback['relevance'] * relevance_weight +
              feedback['clarity'] * clarity_weight +
              feedback['originality'] * originality_weight +
              feedback['engagement'] * engagement_weight)
    return reward

# Fine-tune the model based on feedback
def fine_tune_model(tweets, rewards):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()
    model.train()

    for tweet, reward in zip(tweets, rewards):
        inputs = tokenize_tweet(tweet)
        labels = inputs.clone()

        # Forward pass
        outputs = model(inputs, labels=labels)
        loss = outputs.loss

        # Scale loss by reward
        scaled_loss = loss * reward

        # Backpropagation
        optimizer.zero_grad()
        scaled_loss.backward()
        optimizer.step()

# Streamlit app layout
st.title("AI Tweet Generator with Human Feedback")

# User input for tweet generation prompt
prompt = st.text_input("Enter a prompt for generating a tweet:", "Advice for entrepreneurs")

# Generate button
if st.button("Generate Tweet"):
    new_tweet = generate_tweet(prompt)
    st.write(f"Generated Tweet: {new_tweet}")

    # Collect human feedback
    st.subheader("Provide Feedback on the Generated Tweet:")
    relevance = st.slider("Relevance", 0, 10, 5)
    clarity = st.slider("Clarity", 0, 10, 5)
    originality = st.slider("Originality", 0, 10, 5)
    engagement = st.slider("Engagement", 0, 10, 5)

    # Feedback submission
    if st.button("Submit Feedback"):
        feedback = {
            "relevance": relevance,
            "clarity": clarity,
            "originality": originality,
            "engagement": engagement,
        }
        reward = compute_reward(feedback)
        st.write(f"Feedback submitted! Reward: {reward}")

        # Fine-tune the model based on feedback
        fine_tune_model([new_tweet], [reward])
        st.write("Model fine-tuned based on your feedback!")
