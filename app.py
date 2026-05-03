import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
from dotenv import load_dotenv

# load env
load_dotenv() # Loads variables from .env
os.environ["GROQ_API_KEY"]

# Page configuration
st.set_page_config(
    page_title="Urdu Reasoning AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Urdu text and styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Urdu Typesetting', serif;
        font-size: 1.3rem;
        line-height: 2.5;
        direction: rtl;
        text-align: right;
    }
    
    .thinking-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #00d4ff;
        color: white;
    }
    
    .answer-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #00ff88;
        color: white;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        direction: rtl; 
        font-family: 'Noto Nastaliq Urdu', serif;
        margin-bottom: 10px;
        
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    .model-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 2px;
    }
    
    .badge-qwen { background: #ff6b6b; color: white; }
    .badge-groq { background: #4ecdc4; color: white; }
    .badge-urdu { background: #45b7d1; color: white; }
    
    .thinking-step {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 10px 15px;
        margin: 8px 0;
        border-right: 3px solid #ffd700;
    }
    
    .step-number {
        background: #ffd700;
        color: #1e3c72;
        border-radius: 50%;
        width: 25px;
        height: 25px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.8rem;
        margin-left: 10px;
    }
    
    .urdu-input textarea {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 1.2rem;
    }
    
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.2);
        overflow: hidden;
        margin-top: 5px;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #ffd700, #ff6b6b);
        transition: width 1s ease;
    }
    
    .example-card {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .example-card:hover {
        background: rgba(102, 126, 234, 0.2);
        transform: translateX(5px);
    }
    
    .timer-display {
        font-family: monospace;
        font-size: 1.2rem;
        color: #00d4ff;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 15px 0;
    }
    
    .stat-box {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00d4ff;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.7);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'model_name' not in st.session_state:
    st.session_state.model_name = "qwen-2.5-32b"
if 'reasoning_depth' not in st.session_state:
    st.session_state.reasoning_depth = "deep"
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0
if 'total_time' not in st.session_state:
    st.session_state.total_time = 0

# Sidebar configuration
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ ترتیبات</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Settings</p>", unsafe_allow_html=True)
    
    # API Key input
    # api_key = st.text_input(
    #     "🔑 Groq API Key",
    #     type="password",
    #     value=st.session_state.api_key,
    #     placeholder="gsk_...",
    #     help="Enter your Groq API key"
    # )
    # st.session_state.api_key = api_key
    st.session_state.api_key = os.environ.get("GROQ_API_KEY", "")
    
    st.divider()
    
    # Model selection
    model_options = {
        "Aqal 1.0 1.5B": "llama-3.3-70b-versatile"
    }
    
    selected_model = st.selectbox(
        "🤖 Select Model",
        options=list(model_options.keys()),
        index=0
    )
    st.session_state.model_name = model_options[selected_model]
    
    # Reasoning depth
    reasoning_options = {
        "Quick (تیز)": "quick",
        "Standard (معیاری)": "standard", 
        "Deep (گہری)": "deep"
    }
    
    selected_reasoning = st.selectbox(
        "🧠 Reasoning Depth",
        options=list(reasoning_options.keys()),
        index=2
    )
    st.session_state.reasoning_depth = reasoning_options[selected_reasoning]
    
    st.divider()
    
    # Stats
    st.markdown("<h3>📊 اعداد و شمار</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: gray;'>Statistics</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Questions", st.session_state.total_questions)
    with col2:
        avg_time = st.session_state.total_time / max(st.session_state.total_questions, 1)
        st.metric("Avg Time", f"{avg_time:.1f}s")
    with col3:
        st.metric("Model", selected_model.split()[1] if len(selected_model.split()) > 1 else "Qwen")
    
    st.divider()
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    

# Main content
st.markdown("<h1 class='main-header'>🧠 Aqal First Urdu Reasoning Large Language Model</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; margin-bottom: 30px;'>Urdu Reasoning AI System with Visible Thinking Process</p>", unsafe_allow_html=True)



# Function to initialize LLM
def get_llm():
    if not st.session_state.api_key:
        return None
    return ChatGroq(
        groq_api_key=st.session_state.api_key,
        model_name=st.session_state.model_name,
        temperature=0.3,
        max_tokens=4096,
        streaming=True
    )

# Function to generate reasoning prompt
def get_reasoning_prompt(question: str, depth: str = "deep"):
    depth_instructions = {
        "quick": "Provide brief 2-3 step reasoning.",
        "standard": "Provide 4-5 step detailed reasoning.",
        "deep": "Provide comprehensive step-by-step reasoning with analysis of all possibilities."
    }
    
    system_prompt = f"""You are an advanced Urdu reasoning AI assistant and your name is Aqal(عقل). Your task is to:
1. Think through the problem step-by-step in English (for clarity)
2. Provide the final answer in Urdu
3. Show your complete reasoning process

Reasoning Depth: {depth_instructions.get(depth, depth_instructions['deep'])}

Format your response EXACTLY as follows:

<THINKING>
[Your detailed step-by-step reasoning process in English]
Step 1: [Analysis]
Step 2: [Analysis]
...
Confidence: [X]%
</THINKING>

<ANSWER>
[Your final answer in proper Urdu with not full explanation]
</ANSWER>

Important:
- Always use proper Urdu script (اردو) in the ANSWER section
- Use formal Urdu (فصیح اردو)
- Include examples where relevant
- Make the reasoning transparent and educational
"""
    
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content=question)
    ])

# Function to parse thinking and answer
def parse_response(response: str):
    thinking = ""
    answer = ""
    confidence = 0
    
    # Extract thinking
    think_start = response.find("<THINKING>")
    think_end = response.find("</THINKING>")
    if think_start != -1 and think_end != -1:
        thinking = response[think_start + 10:think_end].strip()
    
    # Extract answer
    ans_start = response.find("<ANSWER>")
    ans_end = response.find("</ANSWER>")
    if ans_start != -1 and ans_end != -1:
        answer = response[ans_start + 8:ans_end].strip()
    
    # Extract confidence
    conf_match = thinking.find("Confidence:")
    if conf_match != -1:
        conf_str = thinking[conf_match:conf_match + 15]
        try:
            confidence = int(''.join(filter(str.isdigit, conf_str)))
        except:
            confidence = 85
    
    # If parsing fails, use fallback
    if not thinking and not answer:
        parts = response.split("Answer:", 1)
        if len(parts) == 2:
            thinking = parts[0].replace("Thinking:", "").strip()
            answer = parts[1].strip()
        else:
            answer = response
            thinking = "Reasoning process not clearly separated."
    
    return thinking, answer, confidence

# Function to format thinking steps
def format_thinking(thinking_text: str):
    steps = []
    lines = thinking_text.split('\n')
    current_step = ""
    
    for line in lines:
        line = line.strip()
        if line.startswith("Step") or line.startswith("**Step"):
            if current_step:
                steps.append(current_step)
            current_step = line
        elif line and not line.startswith("Confidence"):
            current_step += " " + line
    
    if current_step:
        steps.append(current_step)
    
    if not steps:
        steps = [thinking_text]
    
    return steps

# Example questions
st.markdown("<h3 style='margin-top: 20px;'>📋 نمونہ سوالات / Example Questions</h3>", unsafe_allow_html=True)

example_questions = [
    "اگر ایک ٹرین 60 کلومیٹر فی گھنٹہ کی رفتار سے چل رہی ہے اور دوسری ٹرین اس کے مخالف سمت میں 40 کلومیٹر فی گھنٹہ سے چل رہی ہے، تو وہ ایک دوسرے کو کب ملیں گی؟",
    "ایک باغ میں 10 درخت ہیں۔ ہر درخت پر 5 شاخیں ہیں اور ہر شاخ پر 3 پھل ہیں۔ کل کتنے پھل ہیں؟",
    "اگر آج بدھ ہے تو 100 دن بعد کون سا دن ہوگا؟",
    "ایک عدد ایسا بتائیں جو 3 سے تقسیم کرنے پر 1 بچے، 5 سے تقسیم کرنے پر 3 بچیں، اور 7 سے تقسیم کرنے پر 5 بچیں۔",
    "اگر ali کے پاس 500 روپے ہیں اور وہ 3 کتابیں 120 روپے کی شرح سے خریدتا ہے، تو اسے کتنے روپے واپس ملیں گے؟"
]

cols = st.columns(len(example_questions))
for i, (col, q) in enumerate(zip(cols, example_questions)):
    with col:
        if st.button(f"Q{q}", key=f"ex_{i}", help=q[:50] + "..."):
            st.session_state.current_question = q
            st.rerun()

# Input section
st.markdown("<h3 style='margin-top: 30px;'>✍️ اپنا سوال درج کریں / Enter Your Question</h3>", unsafe_allow_html=True)

# Use example question if selected
if 'current_question' in st.session_state:
    default_question = st.session_state.current_question
    st.session_state.current_question
else:
    default_question = ""

question = st.text_area(
    "",
    value=default_question,
    height=120,
    placeholder="یہاں اپنا اردو سوال ٹائپ کریں...\nType your Urdu reasoning question here...",
    key="question_input",
    label_visibility="collapsed"
)

# Add RTL class to the text area
st.markdown("<style>div[data-testid='stTextArea'] textarea { direction: rtl; text-align: right; font-family: 'Noto Nastaliq Urdu', serif; font-size: 1.2rem; }</style>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit_button = st.button("🚀 حل کریں / Solve", use_container_width=True, type="primary")

# Process question
if submit_button and question.strip():
    if not st.session_state.api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar first!")
        st.info("Get your API key from: https://console.groq.com/keys")
    else:
        import time
        start_time = time.time()
        
        # Update stats
        st.session_state.total_questions += 1
        
        # Create containers for streaming
        thinking_container = st.container()
        answer_container = st.container()
        
        with st.spinner("🧠 سوچ رہا ہے... Thinking..."):
            try:
                llm = get_llm()
                if not llm:
                    st.error("Failed to initialize LLM")
                    st.stop()
                
                # Prepare prompt
                prompt = get_reasoning_prompt(question, st.session_state.reasoning_depth)
                
                # Format history
                history_messages = []
                for msg in st.session_state.chat_history[-4:]:  # Last 4 messages for context
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        history_messages.append(AIMessage(content=msg["content"]))
                
                # Generate response
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({"history": history_messages})
                
                # Parse response
                thinking, answer, confidence = parse_response(response)
                
                # Update time stats
                elapsed = time.time() - start_time
                st.session_state.total_time += elapsed
                
                # Save to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question,
                    "thinking": thinking,
                    "answer": answer,
                    "confidence": confidence,
                    "time": elapsed,
                    "model": st.session_state.model_name
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Thinking: {thinking}\n\nAnswer: {answer}"
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()
        
        # Display Thinking Process
        with thinking_container:
            st.markdown("""
            <div class="thinking-box">
                <h3 style="color: #ffd700; margin-bottom: 15px;">🧠 عملِ استدلال / Thinking Process</h3>
            """, unsafe_allow_html=True)
            
            steps = format_thinking(thinking)
            for i, step in enumerate(steps, 1):
                st.markdown(f"""
                <div class="thinking-step">
                    <span class="step-number">{i}</span>
                    <span style="font-size: 0.95rem;">{step}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence bar
            st.markdown(f"""
                <div style="margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.9rem;">Confidence / یقین دہانی:</span>
                        <span style="font-weight: bold; color: #ffd700;">{confidence}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display Answer
        with answer_container:
            st.markdown("""
            <div class="answer-box">
                <h3 style="color: #00ff88; margin-bottom: 15px;">✅ جواب / Answer</h3>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="urdu-text">
                {answer}
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📋 Copy Answer", key="copy_ans"):
                    st.toast("Answer copied to clipboard!")
            with col2:
                if st.button("🔍 Explain More", key="explain_more"):
                    st.session_state.current_question = f"مزید وضاحت کریں: {question}"
                    st.rerun()
            with col3:
                if st.button("❓ Related Question", key="related"):
                    st.session_state.current_question = f"اس سے متعلق ایک اور سوال: {question}"
                    st.rerun()

# Display chat history
if st.session_state.chat_history:
    st.markdown("<h3 style='margin-top: 40px;'>📜 تاریخچہ / History</h3>", unsafe_allow_html=True)
    
    for i in range(0, len(st.session_state.chat_history), 2):
        if i < len(st.session_state.chat_history):
            user_msg = st.session_state.chat_history[i]
            if i + 1 < len(st.session_state.chat_history):
                # This is a completed Q&A pair
                with st.expander(f"Q{(i//2)+1}: {user_msg['content'][:60]}...", expanded=False):
                    st.markdown(f"**Question:** {user_msg['content']}")
                    if 'thinking' in user_msg:
                        st.markdown("**Thinking:**")
                        st.code(user_msg['thinking'], language="text")
                    if 'answer' in user_msg:
                        st.markdown("**Answer:**")
                        st.markdown(f"<div class='urdu-text'>{user_msg['answer']}</div>", unsafe_allow_html=True)
                    if 'confidence' in user_msg:
                        st.progress(user_msg['confidence'] / 100, text=f"Confidence: {user_msg['confidence']}%")

# Footer
# st.markdown("---")
# st.markdown("""
# <div style="text-align: center; padding: 20px; color: gray;">
#     <p>🧠 <strong>Urdu Reasoning AI</strong> | Built with Streamlit + LangChain + Groq + Qwen</p>
#     <p style="font-size: 0.8rem;">Supports complex logical, mathematical, and linguistic reasoning in Urdu</p>
# </div>
# """, unsafe_allow_html=True)
