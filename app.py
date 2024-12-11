import streamlit as st
import openai
from google.cloud import aiplatform
import os
import json

# API 키 설정
openai.api_key = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]

def get_chatgpt_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant that helps answer questions about SKMS using the 4O model (Objective, Obstacle, Options, Output)."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ChatGPT Error: {str(e)}"

def get_gemini_response(prompt):
    try:
        aiplatform.init(project=st.secrets["GOOGLE_CLOUD_PROJECT"])
        
        model = aiplatform.Model("models/text-bison@001")
        response = model.predict(prompt)
        return response.predictions[0]
    except Exception as e:
        return f"Gemini Error: {str(e)}"

def get_claude_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="claude-3-sonnet",
            messages=[
                {"role": "system", "content": "You are Claude 3.5 Sonnet. Your role is to analyze responses from ChatGPT and Gemini Pro, then provide a comprehensive synthesis."},
                {"role": "user", "content": f"Analyze and synthesize these responses about SKMS:\n\nChatGPT: {chatgpt_response}\n\nGemini: {gemini_response}\n\nOriginal question: {prompt}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Claude Error: {str(e)}"

# Streamlit UI
st.title("SK Management System (SKMS) AI Assistant")

# User input
user_prompt = st.text_input("Ask a question about SKMS:")

if user_prompt:
    # Get responses from all models
    with st.spinner("Getting responses from AI models..."):
        chatgpt_response = get_chatgpt_response(user_prompt)
        gemini_response = get_gemini_response(user_prompt)
        
        # Get Claude's synthesis
        claude_synthesis = get_claude_response(user_prompt)
    
    # Display individual responses
    with st.expander("View Individual AI Responses"):
        st.subheader("ChatGPT Response (4O Model)")
        st.write(chatgpt_response)
        
        st.subheader("Gemini Pro Response")
        st.write(gemini_response)
    
    # Display final synthesis
    st.subheader("Final Synthesis by Claude")
    st.write(claude_synthesis) 