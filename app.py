import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

# Streamlit 페이지 설정
st.set_page_config(page_title="SKMS AI Assistant", layout="wide")

# SKMS 문서 로드
with open('skms.txt', 'r', encoding='utf-8') as file:
    SKMS_CONTENT = file.read()

# API 클라이언트 초기화
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def stream_chatgpt_response(prompt, placeholder):
    try:
        message = ""
        stream = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """당신은 SKMS 전문가입니다. 
                반드시 주어진 SKMS 문서의 내용만을 기반으로 답변해주세요. 
                SKMS에 명시되지 않은 일반적인 내용은 포함하지 마세요.
                답변의 각 내용에 대해 SKMS의 어느 부분을 참고했는지 명시해주세요."""},
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
            system="""당신은 SKMS 전문가입니다. 
            반드시 주어진 SKMS 문서의 내용만을 기반으로 답변해주세요. 
            SKMS에 명시되지 않은 일반적인 내용은 포함하지 마세요.
            답변의 각 내용에 대해 SKMS의 어느 부분을 참고했는지 명시해주세요.""",
            messages=[{
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

def stream_gemini_response(prompt, placeholder):
    try:
        message = ""
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            f"""당신은 SKMS 전문가입니다. 
            반드시 아래 주어진 SKMS 문서의 내용만을 기반으로 답변해주세요. 
            SKMS에 명시되지 않은 일반적인 내용은 포함하지 마세요.
            답변의 각 내용에 대해 SKMS의 어느 부분을 참고했는지 명시해주세요.
            
            SKMS: {SKMS_CONTENT}
            
            질문: {prompt}""",
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                message += chunk.text
                placeholder.markdown(message + "▌")
        placeholder.markdown(message)
        return message
    except Exception as e:
        placeholder.error(f"Gemini Error: {str(e)}")
        return f"Gemini Error: {str(e)}"

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
st.title("SKMS AI Assistant")
st.write("SKMS의 내용을 기반으로 AI가 답변해드립니다.")

# 사용자 입력
user_prompt = st.text_input("질문을 입력하세요:", key="user_input")

if user_prompt:
    # ChatGPT 응답
    st.subheader("ChatGPT 응답 (4O 모델)")
    chatgpt_placeholder = st.empty()
    chatgpt_response = stream_chatgpt_response(user_prompt, chatgpt_placeholder)
    
    # 구분선 추가
    st.markdown("---")
    
    # Claude 응답
    st.subheader("Claude 응답")
    claude_placeholder = st.empty()
    claude_response = stream_claude_response(user_prompt, claude_placeholder)
    
    # 구분선 추가
    st.markdown("---")
    
    # Gemini 응답
    st.subheader("Gemini 응답")
    gemini_placeholder = st.empty()
    gemini_response = stream_gemini_response(user_prompt, gemini_placeholder)
    
    # 구분선 추가
    st.markdown("---")
    
    # 종합 분석
    st.subheader("종합 분석")
    synthesis_placeholder = st.empty()
    
    synthesis_prompt = f"""
    원본 질문: {user_prompt}

    ChatGPT의 답변: {chatgpt_response}
    
    Claude의 답변: {claude_response}
    
    Gemini의 답변: {gemini_response}
    """
    
    get_final_synthesis(synthesis_prompt, synthesis_placeholder) 