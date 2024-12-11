import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from google.cloud import aiplatform
import os

# Streamlit 페이지 설정
st.set_page_config(page_title="SKMS AI Assistant", layout="wide")

# API 키라이언트 초기화
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# 각 AI 모델의 응답을 가져오는 함수들
def get_chatgpt_response(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "SKMS 관련 질문에 4O 모델(Objective, Obstacle, Options, Output)을 사용하여 답변해주세요."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ChatGPT Error: {str(e)}"

def get_claude_response(prompt):
    try:
        message = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # content가 리스트이고 첫 번째 요소가 있는지 확인
        if message.content and len(message.content) > 0:
            # text 속성이 있는지 확인
            if hasattr(message.content[0], 'text'):
                return message.content[0].text
            # 직접 텍스트 값에 접근 시도
            elif isinstance(message.content[0], dict) and 'text' in message.content[0]:
                return message.content[0]['text']
        
        return "응답을 처리할 수 없습니다."
    except Exception as e:
        return f"Claude Error: {str(e)}"

def get_gemini_response(prompt):
    try:
        aiplatform.init(project=st.secrets["GOOGLE_CLOUD_PROJECT"])
        model = aiplatform.Model("models/text-bison@001")
        response = model.predict(prompt)
        return response.predictions[0]
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# Streamlit UI
st.title("SK Management System (SKMS) AI Assistant")
st.write("SKMS에 대해 궁금한 을 질문해주세요.")

# 사용자 입력
user_prompt = st.text_input("질문을 입력하세요:", key="user_input")

if user_prompt:
    with st.spinner("AI 모델들이 응답을 생성중입니다..."):
        # 각 모델의 응답 가져오기
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("ChatGPT 응답")
            chatgpt_response = get_chatgpt_response(user_prompt)
            st.write(chatgpt_response)
            
        with col2:
            st.subheader("Claude 응답")
            claude_response = get_claude_response(user_prompt)
            st.write(claude_response)
            
        with col3:
            st.subheader("Gemini 응답")
            gemini_response = get_gemini_response(user_prompt)
            st.write(gemini_response)
        
        # 종합 분석
        st.subheader("종합 분석")
        synthesis_prompt = f"""
        다음 세 AI의 SKMS 관련 응답을 분석하고 종합해주세요:
        
        ChatGPT: {chatgpt_response}
        Claude: {claude_response}
        Gemini: {gemini_response}
        
        원본 질문: {user_prompt}
        """
        
        final_synthesis = get_claude_response(synthesis_prompt)
        st.write(final_synthesis) 