import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import time

# Streamlit 페이지 설정
st.set_page_config(page_title="SKMS AI Assistant", layout="wide")

# SKMS 문서 로드
with open('skms.txt', 'r', encoding='utf-8') as file:
    SKMS_CONTENT = file.read()

# API 클라이언트 초기화
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

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
                messages=[
                    {"role": "system", "content": """당신은 따뜻하고 공감적인 SKMS 전문가입니다. 
                    질문자의 고민에 깊이 공감하면서, SKMS의 경영철학과 가치를 기반으로 답변해주세요.
                    답변 시에는 친근하고 이해하기 쉬운 표현을 사용하되, 전문성은 유지해주세요.
                    질문에 대한 직접적인 내용이 SKMS에 없더라도, SKMS의 경영철학과 핵심가치를 바탕으로 
                    건설적이고 희망적인 관점에서 답변을 제공해주세요.
                    답변 시 참고한 SKMS의 관련 내용이나 철학을 자연스럽게 연결하여 설명해주세요."""},
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
    retries = 1
    for attempt in range(retries):
        try:
            message = ""
            with anthropic_client.messages.stream(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                system="""당신은 따뜻하고 공감적인 SKMS 전문가입니다. 
                질문자의 고민에 깊이 공감하면서, SKMS의 경영철학과 가치를 기반으로 답변해주세요.
                답변 시에는 친근하고 이해하기 쉬운 표현을 사용하되, 전문성은 유지해주세요.
                질문에 대한 직접적인 내용이 SKMS에 없더라도, SKMS의 경영철학과 핵심가치를 바탕으로 
                건설적이고 희망적인 관점에서 답변을 제공해주세요.
                답변 시 참고한 SKMS의 관련 내용이나 철학을 자연스럽게 연결하여 설명해주세요.""",
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
            error_message = str(e)
            if "503" in error_message:
                error_display = "Anthropic 서버가 일시적으로 응답하지 않습니다 (503 Service Unavailable)"
            elif "timeout" in error_message.lower():
                error_display = "Anthropic 서버 응답 시간 초과"
            else:
                error_display = f"Anthropic 서버 오류: {error_message}"
            
            if attempt < retries:
                placeholder.warning(f"{error_display}... 재시도 중")
                time.sleep(2)
                continue
            placeholder.error(error_display)
            return f"Claude Error: {error_display}"

def stream_gemini_response(prompt, placeholder):
    retries = 1
    for attempt in range(retries):
        try:
            message = ""
            # 안정적인 구성을 위한 설정
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

            prompt_parts = [
                """당신은 따뜻하고 공감적인 SKMS 전문가입니다. 
                질문자의 고민에 깊이 공감하면서, SKMS의 경영철학과 가치를 기반으로 답변해주세요.
                답변 시에는 친근하고 이해하기 쉬운 표현을 사용하되, 전문성은 유지해주세요.
                질문에 대한 직접적인 내용이 SKMS에 없더라도, SKMS의 경영철학과 핵심가치를 바탕으로 
                건설적이고 희망적인 관점에서 답변을 제공해주세요.
                답변 시 참고한 SKMS의 관련 내용이나 철학을 자연스럽게 연결하여 설명해주세요.""",
                f"\nSKMS: {SKMS_CONTENT}",
                f"\n질문: {prompt}"
            ]

            response = model.generate_content(
                prompt_parts,
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

def get_final_synthesis(prompt, placeholder):
    try:
        message = ""
        with anthropic_client.messages.stream(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            system="""당신은 SKMS 전문가입니다.
            각 AI의 답변들을 종합하여 SKMS의 관점에서 가장 핵심적이고 통찰력 있는 답변을 제시해주세요.
            답변 시 SKMS의 철학과 가치를 자연스럽게 연결하여 설명해주세요.
            
            반드시 지켜야 할 규칙:
            1. 최소 2개 이상의 다른 응답들과 공통되는 관점을 중심으로 답변하세요.
            2. SKMS 원문의 내용과 모순되지 않도록 답변하세요.
            
            여러 AI의 답변 중 중복되는 중요한 관점이나, 서로 보완되는 관점들을 잘 통합해서 설명해주세요.""",
            messages=[{
                "role": "user",
                "content": f"""
                원본 질문: {prompt}

                ChatGPT의 답변: {chatgpt_response}
                
                Claude의 답변: {claude_response}
                
                Gemini의 답변: {gemini_response}
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
st.write("SKMS를 기반으로 AI가 답변해드립니다. 여러분의 고민을 말씀해 주세요.")

# 세션 상태 초기화 (필요한 경우)
if 'previous_input' not in st.session_state:
    st.session_state.previous_input = ""

# 사용자 입력과 버튼
user_prompt = st.text_area("", height=100, key="user_input")

if st.button("답변 생성하기"):
    if user_prompt:
        if user_prompt != st.session_state.previous_input:
            st.session_state.previous_input = user_prompt
            
            # ChatGPT 답변
            st.markdown('<div class="llm-header">ChatGPT 답변</div>', unsafe_allow_html=True)
            chatgpt_placeholder = st.empty()
            chatgpt_response = stream_chatgpt_response(user_prompt, chatgpt_placeholder)
            st.markdown('<div class="response-divider"></div>', unsafe_allow_html=True)
            
            # Claude 답변
            st.markdown('<div class="llm-header">Claude 답변</div>', unsafe_allow_html=True)
            claude_placeholder = st.empty()
            claude_response = stream_claude_response(user_prompt, claude_placeholder)
            st.markdown('<div class="response-divider"></div>', unsafe_allow_html=True)
            
            # Gemini 답변
            st.markdown('<div class="llm-header">Gemini 답변</div>', unsafe_allow_html=True)
            gemini_placeholder = st.empty()
            gemini_response = stream_gemini_response(user_prompt, gemini_placeholder)
            st.markdown('<div class="response-divider"></div>', unsafe_allow_html=True)
            
            # 종합 답변
            st.markdown('<div class="llm-header">종합 답변</div>', unsafe_allow_html=True)
            synthesis_placeholder = st.empty()
            
            synthesis_prompt = f"""
            원본 질문: {user_prompt}

            ChatGPT의 답변: {chatgpt_response}
            
            Claude의 답변: {claude_response}
            
            Gemini의 답변: {gemini_response}
            """
            
            get_final_synthesis(synthesis_prompt, synthesis_placeholder) 