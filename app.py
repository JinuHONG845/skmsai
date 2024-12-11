import streamlit as st
from openai import OpenAI
from anthropic import Anthropic

# Streamlit 페이지 설정
st.set_page_config(page_title="SKMS AI Assistant", layout="wide")

# SKMS 문서 로드
with open('skms.txt', 'r', encoding='utf-8') as file:
    SKMS_CONTENT = file.read()

# API 클라이언트 초기화
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

def stream_chatgpt_response(prompt, placeholder):
    try:
        message = ""
        stream = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 SKMS 전문가입니다. 아래의 SKMS를 기반으로 질문에 답변해주세요. 4O 모델(Objective, Obstacle, Options, Output)을 사용하여 답변해주세요."},
                {"role": "user", "content": f"SKMS: {SKMS_CONTENT}\n\n질문: {prompt}"}
            ],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                message += chunk.choices[0].delta.content
                placeholder.markdown(message + "▌")
        placeholder.markdown(message)
        return message
    except Exception as e:
        placeholder.error(f"ChatGPT Error: {str(e)}")
        return f"ChatGPT Error: {str(e)}"

def stream_claude_response(prompt, placeholder):
    try:
        message = ""
        with anthropic_client.messages.stream(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{
                "role": "system",
                "content": "당신은 SKMS 전문가입니다. 아래의 SKMS를 기반으로 질문에 답변해주세요."
            }, {
                "role": "user",
                "content": f"SKMS: {SKMS_CONTENT}\n\n질문: {prompt}"
            }]
        ) as stream:
            for text in stream.text_stream:
                message += text
                placeholder.markdown(message + "▌")
        placeholder.markdown(message)
        return message
    except Exception as e:
        placeholder.error(f"Claude Error: {str(e)}")
        return f"Claude Error: {str(e)}"

def get_final_synthesis(prompt, placeholder):
    try:
        message = ""
        with anthropic_client.messages.stream(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""
                SKMS: {SKMS_CONTENT}

                다음 답변들을 종합하여 가장 정확하고 실용적인 최종 답변을 제시해주세요:
                {prompt}
                
                답변 형식:
                1. 핵심 요약
                2. 주요 시사점
                3. 실행 방안
                """
            }]
        ) as stream:
            for text in stream.text_stream:
                message += text
                placeholder.markdown(message + "▌")
        placeholder.markdown(message)
        return message
    except Exception as e:
        placeholder.error(f"Synthesis Error: {str(e)}")
        return f"Synthesis Error: {str(e)}"

# Streamlit UI
st.title("SK Management System (SKMS) AI Assistant")
st.write("SKMS에 대해 궁금한 점을 질문해주세요.")

# 사용자 입력
user_prompt = st.text_input("질문을 입력하세요:", key="user_input")

if user_prompt:
    # 각 모델의 응답을 위한 컬럼 생성
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ChatGPT 응답 (4O 모델)")
        chatgpt_placeholder = st.empty()
        
    with col2:
        st.subheader("Claude 응답")
        claude_placeholder = st.empty()
    
    # 응답 생성
    chatgpt_response = stream_chatgpt_response(user_prompt, chatgpt_placeholder)
    claude_response = stream_claude_response(user_prompt, claude_placeholder)
    
    # 구분선 추가
    st.markdown("---")
    
    # 종합 분석
    st.subheader("종합 분석")
    synthesis_placeholder = st.empty()
    
    synthesis_prompt = f"""
    원본 질문: {user_prompt}

    ChatGPT의 답변: {chatgpt_response}
    
    Claude의 답변: {claude_response}
    """
    
    get_final_synthesis(synthesis_prompt, synthesis_placeholder) 