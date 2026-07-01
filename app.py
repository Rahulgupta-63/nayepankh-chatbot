import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="NayePankh AI Assistant",
    page_icon="🕊️",
    layout="centered"
)

# ============================================
# STYLING
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #f0f7f4;
    color: #1a2e1a;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #1a4a2e;
    border-right: none;
}

[data-testid="stSidebar"] * {
    color: #e8f5e9 !important;
}

h1 {
    font-family: 'Playfair Display', serif;
    font-weight: 800;
    color: #1a4a2e;
}

.hero-tag {
    background: #1a4a2e;
    color: #e8f5e9;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    display: inline-block;
    margin-bottom: 8px;
}

.chat-user {
    background: #ffffff;
    border-left: 3px solid #1a4a2e;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 0.9rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.chat-assistant {
    background: #e8f5e9;
    border-left: 3px solid #4caf50;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 0.9rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.status-box {
    background: rgba(76, 175, 80, 0.1);
    border: 1px solid #4caf50;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #1a4a2e;
    font-weight: 600;
}

.impact-box {
    background: #1a4a2e;
    color: #e8f5e9;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    text-align: center;
}

.impact-number {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 800;
    color: #81c784;
}

.impact-label {
    font-size: 0.75rem;
    opacity: 0.8;
}

.stButton > button {
    background: #1a4a2e;
    color: #e8f5e9;
    border: none;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
}

.stButton > button:hover {
    background: #2e7d32;
    color: #ffffff;
}

[data-testid="stTextInput"] input {
    background: #ffffff;
    border: 1px solid #c8e6c9;
    color: #1a2e1a;
    font-family: 'Inter', sans-serif;
    border-radius: 6px;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #4caf50;
    border-radius: 8px;
}

.suggested-question {
    background: #ffffff;
    border: 1px solid #c8e6c9;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.85rem;
    cursor: pointer;
    margin: 4px 0;
    color: #1a4a2e;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "notes_loaded" not in st.session_state:
    st.session_state.notes_loaded = False

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found.")
        st.stop()
    return Groq(api_key=api_key)

# ============================================
# KNOWLEDGE BASE — preloaded NayePankh content
# ============================================
NAYEPANKH_KNOWLEDGE = [
    "NayePankh Foundation is a UP Government registered NGO founded on 28th March 2021. It is one of the biggest student-led organizations in India, working towards uplifting the underprivileged. The foundation is registered under the Indian Society Act 1860, affiliated with NITI Aayog, and holds 80G and 12A certifications.",
    "NayePankh Foundation was started during the COVID-19 pandemic in 2020 by a group of high school students who wanted to help the underprivileged. They started by distributing food and essential supplies during the lockdown. The foundation officially launched on 28th March 2021.",
    "NayePankh means new wings — the foundation aims to give wings to the underprivileged sections of society. Their motto is BADALTE BHARAT KI NAYI TASVEER which means A new picture of changing India.",
    "NayePankh Foundation distributes free food to underprivileged people and stray animals to fight hunger. They also provide free sanitary napkins to women and run awareness campaigns about personal hygiene among women and youth.",
    "NayePankh Foundation provides clothes to poor families and works towards educating underprivileged sectors of society. They run awareness campaigns on personal hygiene, menstrual health, and community empowerment.",
    "NayePankh Foundation has helped more than 2 lakh which is 200,000 underprivileged people till date. They have trained and supported more than 30,000 interns. They operate primarily in Kanpur and Ghaziabad, Uttar Pradesh.",
    "NayePankh Foundation has been featured in major newspapers including The Pioneer, Dainik Jagran, and Hindustan. They are one of the most recognized student-led NGOs in India.",
    "NayePankh Foundation is 80G and 12A certified. Donors get 50% relief in income tax on their donations. Donations to NayePankh Foundation qualify for tax benefits under Indian law.",
    "To volunteer with NayePankh Foundation, contact them at contact@nayepankh.com or call +91-8318500748. You can also apply through Internshala where they regularly post internship opportunities. They welcome students and young professionals.",
    "You can donate to NayePankh Foundation at nayepankh.com/donate. Every donation makes a difference. Your donation can provide a child with a school uniform, a meal, or access to education.",
    "Contact NayePankh Foundation: Email: contact@nayepankh.com, Mobile: +91-8318500748, Website: nayepankh.com, Instagram: instagram.com/nayepankhfoundation, LinkedIn: linkedin.com/company/nayepankh, Facebook: facebook.com/nayepankhfoundation",
    "NayePankh Foundation welcomes students, young professionals, and anyone who wants to contribute to society. Most of their team members are college and school students. Whether you are passionate about education, health, food distribution, or awareness campaigns, there is a place for you.",
    "NayePankh Foundation was founded with the vision to uplift underprivileged and marginalized communities and provide them access to education, healthcare, and basic necessities. They believe every child has the right to dream regardless of their socio-economic background.",
    "NayePankh Foundation certifications: Registered under Indian Society Act 1860, UP Government registered NGO, Affiliated with NITI Aayog, 80G certified for tax benefit for donors, 12A certified for income tax exemption for the NGO.",
]

# ============================================
# CHROMADB SETUP
# ============================================
def setup_collection(chunks):
    embedding_model = load_embedding_model()
    client = chromadb.Client()
    try:
        client.delete_collection("nayepankh")
    except:
        pass
    collection = client.create_collection("nayepankh")
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"chunk{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection

# ============================================
# RAG FUNCTIONS
# ============================================
def retrieve(question, collection):
    embedding_model = load_embedding_model()
    q_embedding = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[q_embedding], n_results=4)
    return results['documents'][0]

def ask_nayepankh(question, collection):
    groq_client = get_groq_client()
    relevant_chunks = retrieve(question, collection)
    context = "\n".join(relevant_chunks)

    st.session_state.chat_history.append({"role": "user", "content": question})

    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant for NayePankh Foundation, a UP Government registered NGO in India that uplifts underprivileged communities.

Answer questions about the foundation using ONLY the information below. Be warm, helpful, and encouraging. If someone wants to volunteer or donate, give them the contact details.

If the answer is not in the information provided, say: "I don't have that information right now. Please contact us at contact@nayepankh.com or call +91-8318500748."

Foundation Information:
{context}"""
        }
    ] + st.session_state.chat_history

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    answer = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    return answer

# ============================================
# AUTO-LOAD KNOWLEDGE BASE
# ============================================
if not st.session_state.notes_loaded:
    with st.spinner("Loading NayePankh knowledge base..."):
        st.session_state.collection = setup_collection(NAYEPANKH_KNOWLEDGE)
        st.session_state.notes_loaded = True

# ============================================
# UI
# ============================================

# Sidebar
with st.sidebar:
    st.markdown('<div class="impact-box"><div class="impact-number">2L+</div><div class="impact-label">People Helped</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="impact-box"><div class="impact-number">30K+</div><div class="impact-label">Interns Trained</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="impact-box"><div class="impact-number">2021</div><div class="impact-label">Founded</div></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Quick Links**")
    st.markdown("🌐 [Website](https://nayepankh.com)")
    st.markdown("💰 [Donate](https://nayepankh.com/donate)")
    st.markdown("📸 [Instagram](https://instagram.com/nayepankhfoundation)")
    st.markdown("💼 [LinkedIn](https://linkedin.com/company/nayepankh)")
    st.divider()
    st.markdown("📧 contact@nayepankh.com")
    st.markdown("📞 +91-8318500748")
    st.divider()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("*Built by Rahul Gupta*")
    st.markdown("*MCA — KNIT Sultanpur*")

# Main
st.markdown('<div class="hero-tag">🕊️ UP Govt. Registered NGO · 80G Certified</div>', unsafe_allow_html=True)
st.markdown("# NayePankh Foundation")
st.markdown("**Ask me anything about our work, volunteering, donations, or how to get involved.**")
st.divider()

# Suggested questions
if not st.session_state.chat_history:
    st.markdown("**Try asking:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="suggested-question">🤝 How can I volunteer?</div>', unsafe_allow_html=True)
        st.markdown('<div class="suggested-question">💰 How do I donate?</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="suggested-question">🏢 What does NayePankh do?</div>', unsafe_allow_html=True)
        st.markdown('<div class="suggested-question">🧾 Are donations tax deductible?</div>', unsafe_allow_html=True)
    st.divider()

# Chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-assistant">🕊️ {msg["content"]}</div>', unsafe_allow_html=True)

# Input
question = st.text_input("Ask a question...", key="input", placeholder="e.g. How can I volunteer with NayePankh?")

if st.button("Ask") and question:
    with st.spinner("Finding answer..."):
        answer = ask_nayepankh(question, st.session_state.collection)
    st.rerun()
