import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import time
import requests

# Streamlit 페이지 설정
st.set_page_config(page_title="LLM Big5", layout="wide")

# API 클라이언트 초기화
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
deepseek_client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"], 
    base_url="https://api.deepseek.com"
)
grok_client = OpenAI(
    api_key=st.secrets["GROK_API_KEY"],
    base_url="https://api.x.ai/v1"
)

# 상단에 CSS 스타일 추가
st.markdown("""
    <style>
    .llm-header {
        color: #1f77b4;
        font-size: 1.2em;
        font-weight: 600;
        margin: 25px 0 15px 0;
        padding-left: 10px;
        border-left: 4px solid #1f77b4;
    }
    .response-divider {
        margin: 30px 0;
        border: none;
        border-top: 1px solid #e1e4e8;
    }
    /* 페이지 전체 여백과 중앙 정렬 설정 */
    .block-container {
        max-width: 1000px !important;
        margin: 0 auto !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 5% !important;
        padding-right: 5% !important;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    /* 모바일 환경 대응 */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .stTextArea > div > div > textarea {
            width: 100% !important;
        }
    }
    /* 입력창 중앙 정렬 */
    .stTextArea > div {
        width: 100% !important;
        margin: 0 auto !important;
    }
    </style>
""", unsafe_allow_html=True)

def stream_chatgpt_response(prompt, placeholder):
    retries = 1
    for attempt in range(retries):
        try:
            message = ""
            stream = openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    message += chunk.choices[0].delta.content
                    placeholder.markdown(message + "▌")
            placeholder.markdown(message)
            return message
        except Exception as e:
            error_message = str(e)
            if "503" in error_message:
                error_display = "OpenAI 서버가 일시적으로 응답하지 않습니다 (503 Service Unavailable)"
            elif "timeout" in error_message.lower():
                error_display = "OpenAI 서버 응답 시간 초과"
            else:
                error_display = f"OpenAI 서버 오류: {error_message}"
            
            if attempt < retries:
                placeholder.warning(f"{error_display}... 재시도 중")
                time.sleep(2)
                continue
            placeholder.error(error_display)
            return f"ChatGPT Error: {error_display}"

def stream_claude_response(prompt, placeholder):
    retries = 3
    for attempt in range(retries):
        try:
            message = ""
            with anthropic_client.messages.stream(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            ) as stream:
                for text in stream.text_stream:
                    message += text
                    placeholder.markdown(message + "▌")
            placeholder.markdown(message)
            return message
        except Exception as e:
            error_message = str(e)
            if "overloaded" in str(e).lower():
                error_display = "Anthropic 서버가 과부하 상태입니다. 잠시 후 다시 시도합니다."
                wait_time = (attempt + 1) * 3
            elif "503" in error_message:
                error_display = "Anthropic 서버가 일시적으로 응답하지 않습니다 (503 Service Unavailable)"
                wait_time = 2
            elif "timeout" in error_message.lower():
                error_display = "Anthropic 서버 응답 시간 초과"
                wait_time = 2
            else:
                error_display = f"Anthropic 서버 오류: {error_message}"
                wait_time = 2
            
            if attempt < retries - 1:
                placeholder.warning(f"{error_display}... {wait_time}초 후 재시도")
                time.sleep(wait_time)
                continue
            placeholder.error(error_display)
            return f"Claude Error: {error_display}"

def stream_gemini_response(prompt, placeholder):
    retries = 1
    for attempt in range(retries):
        try:
            message = ""
            generation_config = {
                "temperature": 0.9,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 2048,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
            ]

            model = genai.GenerativeModel(
                model_name='gemini-pro',
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            response = model.generate_content(
                prompt,
                stream=True,
            )

            for chunk in response:
                if hasattr(chunk, 'text'):
                    message += chunk.text
                    placeholder.markdown(message + "▌")
                elif hasattr(chunk, 'parts'):
                    for part in chunk.parts:
                        if hasattr(part, 'text'):
                            message += part.text
                            placeholder.markdown(message + "▌")
            
            placeholder.markdown(message)
            return message

        except Exception as e:
            error_message = str(e)
            if "503" in error_message:
                error_display = "Google AI 서버가 일시적으로 응답하지 않습니다 (503 Service Unavailable)"
            elif "timeout" in error_message.lower():
                error_display = "Google AI 서버 응답 시간 초과"
            else:
                error_display = f"Google AI 서버 오류: {error_message}"
            
            if attempt < retries:
                placeholder.warning(f"{error_display}... 재시도 중")
                time.sleep(2)
                continue
            placeholder.error(error_display)
            return f"Gemini Error: {error_display}"

def stream_deepseek_response(prompt, placeholder):
    retries = 1
    for attempt in range(retries):
        try:
            message = ""
            stream = deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    message += chunk.choices[0].delta.content
                    placeholder.markdown(message + "▌")
            placeholder.markdown(message)
            return message
        except Exception as e:
            error_message = str(e)
            if "402" in error_message or "Insufficient Balance" in error_message:
                error_display = "Deepseek API 잔액이 부족합니다. 관리자에게 문의해주세요."
                placeholder.error(error_display)
                return f"Deepseek Error: {error_display}"
            elif "503" in error_message:
                error_display = "Deepseek 서버가 일시적으로 응답하지 않습니다 (503 Service Unavailable)"
            elif "timeout" in error_message.lower():
                error_display = "Deepseek 서버 응답 시간 초과"
            else:
                error_display = f"Deepseek 서버 오류: {error_message}"
            
            if attempt < retries:
                placeholder.warning(f"{error_display}... 재시도 중")
                time.sleep(2)
                continue
            placeholder.error(error_display)
            return f"Deepseek Error: {error_display}"

def stream_grok_response(prompt, placeholder):
    retries = 1
    for attempt in range(retries):
        try:
            message = ""
            stream = grok_client.chat.completions.create(
                model="grok-2-latest",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                response_format={"type": "text"}
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    message += content
                    placeholder.markdown(message + "▌")
            placeholder.markdown(message)
            return message
        except Exception as e:
            error_message = str(e)
            if "503" in error_message:
                error_display = "Grok 서버가 일시적으로 응답하지 않습니다 (503 Service Unavailable)"
            elif "timeout" in error_message.lower():
                error_display = "Grok 서버 응답 시간 초과"
            else:
                error_display = f"Grok 서버 오류: {error_message}"
            
            if attempt < retries:
                placeholder.warning(f"{error_display}... 재시도 중")
                time.sleep(2)
                continue
            placeholder.error(error_display)
            return f"Grok Error: {error_display}"

# Streamlit UI
st.title("LLM Big5 비교 (v.250217)")
st.markdown("""
    <div style='padding: 10px; background-color: #ffebee; border-left: 5px solid #f44336; margin: 10px 0;'>
        <strong style='color: #d32f2f;'>⚠️ 보안 주의사항:</strong>
        <span style='color: #333;'> Deepseek이 포함되어 있으므로 민감한 정보 입력에 주의해 주세요.</span>
    </div>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'previous_input' not in st.session_state:
    st.session_state.previous_input = ""

# 사용자 입력과 버튼
user_prompt = st.text_area("", height=100, key="user_input")

if st.button("답변 생성하기"):
    if user_prompt:
        st.session_state.previous_input = user_prompt
        
        # ChatGPT 답변
        st.markdown('<div class="llm-header">ChatGPT (GPT-4 Turbo Preview) 답변</div>', unsafe_allow_html=True)
        chatgpt_placeholder = st.empty()
        chatgpt_response = stream_chatgpt_response(user_prompt, chatgpt_placeholder)
        st.markdown('<div class="response-divider"></div>', unsafe_allow_html=True)
        
        # Claude 답변
        st.markdown('<div class="llm-header">Claude (Claude 3 Sonnet) 답변</div>', unsafe_allow_html=True)
        claude_placeholder = st.empty()
        claude_response = stream_claude_response(user_prompt, claude_placeholder)
        st.markdown('<div class="response-divider"></div>', unsafe_allow_html=True)
        
        # Gemini 답변
        st.markdown('<div class="llm-header">Gemini (Gemini Pro) 답변</div>', unsafe_allow_html=True)
        gemini_placeholder = st.empty()
        gemini_response = stream_gemini_response(user_prompt, gemini_placeholder)
        st.markdown('<div class="response-divider"></div>', unsafe_allow_html=True)
        
        # Deepseek 답변
        st.markdown('<div class="llm-header">Deepseek (DeepSeek-R1 Reasoner) 답변</div>', unsafe_allow_html=True)
        deepseek_placeholder = st.empty()
        deepseek_response = stream_deepseek_response(user_prompt, deepseek_placeholder)
        st.markdown('<div class="response-divider"></div>', unsafe_allow_html=True)
        
        # Grok 답변
        st.markdown('<div class="llm-header">Grok (Grok-2 Latest) 답변</div>', unsafe_allow_html=True)
        grok_placeholder = st.empty()
        grok_response = stream_grok_response(user_prompt, grok_placeholder)
        st.markdown('<div class="response-divider"></div>', unsafe_allow_html=True) 