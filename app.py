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
                SKMS의 경영철학과 가치를 기반으로 답변해주세요.
                질문에 대한 직접적인 내용이 SKMS에 없더라도, SKMS의 경영철학과 핵심가치를 바탕으로 
                관련된 맥락에서 건설적인 답변을 제공해주세요.
                답변 시 참고한 SKMS의 관련 내용이나 철학을 함께 언급해주세요."""},
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
            SKMS의 경영철학과 가치를 기반으로 답변해주세요.
            질문에 대한 직접적인 내용이 SKMS에 없더라도, SKMS의 경영철학과 핵심가치를 바탕으로 
            관련된 맥락에서 건설적인 답변을 제공해주세요.
            답변 시 참고한 SKMS의 관련 내용이나 철학을 함께 언급해주세요.""",
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
            SKMS의 경영철학과 가치를 기반으로 답변해주세요.
            질문에 대한 직접적인 내용이 SKMS에 없더라도, SKMS의 경영철학과 핵심가치를 바탕으로 
            관련된 맥락에서 건설적인 답변을 제공해주세요.
            답변 시 참고한 SKMS의 관련 내용이나 철학을 함께 언급해주세요.
            
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
            system="""당신은 SKMS 전문가입니다.
            각 AI의 답변들을 종합하여 SKMS의 관점에서 가장 핵심적이고 통찰력 있는 답변을 제시해주세요.
            답변 시 SKMS의 철학과 가치를 자연스럽게 연결하여 설명해주세요.
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
        # 이전 입력과 다른 경우 화면 초기화
        if user_prompt != st.session_state.previous_input:
            st.empty()  # 기존 내용 초기화
            st.title("SKMS AI Assistant")
            st.write("SKMS를 기반으로 AI가 답변해드립니다. 여러분의 고민을 말씀해 주세요.")
            st.text_area("", value=user_prompt, height=100, key="user_input_display")
            st.session_state.previous_input = user_prompt
        
        # ChatGPT 응답
        st.subheader("ChatGPT 응답")
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