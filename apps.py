import streamlit as st

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Smart Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* MAIN BACKGROUND */

.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}

/* MAIN TITLE */

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 700;
    color: white;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    color: #d1d1d1;
    font-size: 20px;
    margin-bottom: 40px;
}

/* LOGIN CARD */

.login-card {
    background: rgba(255,255,255,0.08);
    padding: 35px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* FEATURE CARD */

.feature-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    margin-bottom: 15px;
}

/* USER MESSAGE */

.user-message {
    background: linear-gradient(to right, #0072ff, #00c6ff);
    padding: 15px;
    border-radius: 18px;
    color: white;
    font-size: 17px;
    margin-top: 10px;
    margin-bottom: 10px;
    margin-left: 30%;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* AI MESSAGE */

.ai-message {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 18px;
    color: white;
    font-size: 17px;
    margin-top: 10px;
    margin-bottom: 20px;
    margin-right: 30%;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* RESULT BOX */

.result-box {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    margin-top: 15px;
    margin-bottom: 25px;
    color: white;
    line-height: 2;
    font-size: 17px;
    white-space: pre-wrap;
    backdrop-filter: blur(10px);
}

/* BUTTONS */

.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    font-size: 16px;
    font-weight: 600;
    padding: 12px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.25);
    backdrop-filter: blur(10px);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# GROQ API
# ==========================================

client = Groq(
    api_key="Your Groq API KEY"
)

# ==========================================
# SESSION STATES
# ==========================================

if "users" not in st.session_state:
    st.session_state.users = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# LOGIN PAGE
# ==========================================

if not st.session_state.logged_in:

    st.markdown("""
    <div class="main-title">
    🤖 AI Smart Document Assistant
    </div>

    <div class="subtitle">
    Conversational RAG System with AI-Powered PDF Understanding
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        option = st.selectbox(
            "Choose Option",
            ["Login", "Signup"]
        )

        username = st.text_input("👤 Username")

        password = st.text_input(
            "🔒 Password",
            type="password"
        )

        # LOGIN

        if option == "Login":

            if st.button("🚀 Login"):

                if username in st.session_state.users:

                    if st.session_state.users[username] == password:

                        st.session_state.logged_in = True
                        st.session_state.username = username

                        st.success("Login Successful ✅")
                        st.rerun()

                    else:
                        st.error("Wrong Password ❌")

                else:
                    st.error("User Not Found ❌")

        # SIGNUP

        if option == "Signup":

            if st.button("✨ Create Account"):

                if username in st.session_state.users:

                    st.warning("Username Already Exists ⚠️")

                else:

                    st.session_state.users[username] = password

                    st.success("Account Created Successfully ✅")

        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# MAIN APPLICATION
# ==========================================

else:

    # SIDEBAR

    with st.sidebar:

        st.markdown(f"""
        ## 👋 Welcome
        ### {st.session_state.username}
        """)

        st.markdown("---")

        st.markdown("""
        <div class="feature-card">
        <h4>🚀 Features</h4>

        ✅ Conversational AI Chat <br>
        ✅ PDF Summarization <br>
        ✅ Semantic Search <br>
        ✅ Quiz Generation <br>
        ✅ Interview Questions <br>
        ✅ RAG Architecture <br>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Logout"):

            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.chat_history = []

            st.rerun()

    # TITLE

    st.markdown("""
    <div class="main-title">
    📄 Smart Document Analyzer
    </div>

    <div class="subtitle">
    Upload PDFs • Ask Questions • Generate AI Insights
    </div>
    """, unsafe_allow_html=True)

    # FILE UPLOAD

    st.subheader("📤 Upload Your PDF")

    uploaded_file = st.file_uploader(
        "Drag and Drop Your PDF Here",
        type="pdf"
    )

    # PROCESS PDF

    if uploaded_file is not None:

        pdf_reader = PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

        st.success("PDF Uploaded Successfully ✅")

        # CHUNKING

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_text(text)

        st.info(f"📚 Number of chunks created: {len(chunks)}")

        # EMBEDDINGS

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # VECTOR DATABASE

        vector_store = FAISS.from_texts(
            chunks,
            embedding=embeddings
        )

        st.success("Embeddings Created Successfully ✅")
        st.success("Vector Database Ready ✅")

        # ==========================================
        # PLUS MENU
        # ==========================================

        with st.expander("➕"):

            # SUMMARY

            if st.button("📄 Summarize PDF"):

                with st.spinner("Generating Summary..."):

                    prompt = f"""
                    Summarize the following document clearly.

                    Document:
                    {text[:12000]}
                    """

                    completion = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    summary = completion.choices[0].message.content

                    st.markdown(f"""
                    <div class="result-box">
                    <h3>📄 PDF Summary</h3>
                    {summary}
                    </div>
                    """, unsafe_allow_html=True)

            # MAIN TOPICS

            if st.button("🧠 Explain Main Topics"):

                with st.spinner("Analyzing Topics..."):

                    prompt = f"""
                    Explain the main topics from this document clearly.

                    Document:
                    {text[:12000]}
                    """

                    completion = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    topics = completion.choices[0].message.content

                    st.markdown(f"""
                    <div class="result-box">
                    <h3>🧠 Main Topics</h3>
                    {topics}
                    </div>
                    """, unsafe_allow_html=True)

            # INTERVIEW QUESTIONS

            if st.button("🎯 Generate Interview Questions"):

                with st.spinner("Generating Questions..."):

                    prompt = f"""
                    Generate interview questions from this document.

                    Document:
                    {text[:12000]}
                    """

                    completion = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    interview = completion.choices[0].message.content

                    st.markdown(f"""
                    <div class="result-box">
                    <h3>🎯 Interview Questions</h3>
                    {interview}
                    </div>
                    """, unsafe_allow_html=True)

            # QUIZ GENERATION

            if st.button("🧠 Generate Quiz"):

                with st.spinner("Generating Quiz..."):

                    prompt = f"""
                    Generate 5 multiple choice quiz questions from the document.

                    IMPORTANT:
                    - Put each option on a NEW LINE
                    - Keep proper spacing
                    - Mention correct answer clearly

                    Format exactly like this:

                    1. Question

                    A) Option

                    B) Option

                    C) Option

                    D) Option

                    Correct Answer: A) Option


                    Document:
                    {text[:12000]}
                    """

                    completion = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    quiz = completion.choices[0].message.content

                    # HIGHLIGHT ANSWERS

                    quiz = quiz.replace(
                        "Correct Answer:",
                        "✅ <span style='color:#00ff99; font-weight:bold;'>Correct Answer:</span>"
                    )

                    st.markdown(f"""
                    <div class="result-box">
                    <h3>🧠 Generated Quiz</h3>

                    <div style="
                        line-height:2.2;
                        font-size:17px;
                        white-space:pre-wrap;
                    ">
                    {quiz}
                    </div>

                    </div>
                    """, unsafe_allow_html=True)

        # CHAT INPUT

        question = st.chat_input(
            "💬 Ask anything from the PDF..."
        )

        # QUESTION ANSWERING

        if question:

            with st.spinner("🤖 AI is analyzing document..."):

                docs = vector_store.similarity_search(question)

                context = ""

                for doc in docs:
                    context += doc.page_content

                prompt = f"""
                Answer the question using only the provided context.

                Context:
                {context}

                Question:
                {question}
                """

                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                answer = completion.choices[0].message.content

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

        # DISPLAY CHAT HISTORY

        for chat in st.session_state.chat_history:

            st.markdown(f"""
            <div class="user-message">
            👤 {chat["question"]}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="ai-message">
            🤖 {chat["answer"]}
            </div>
            """, unsafe_allow_html=True)
