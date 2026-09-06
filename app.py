import streamlit as st
import helper
import pickle


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Quora Duplicate Question Detector",
    page_icon="🔍",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a,
            #111827,
            #1e293b
        );
    }

    .block-container {
        max-width: 900px;
        padding-top: 45px;
        padding-bottom: 40px;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: white;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #cbd5e1;
        margin-bottom: 35px;
    }

    .question-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .question-label {
        font-size: 18px;
        font-weight: 700;
        color: white;
        margin-bottom: 10px;
    }

    .stButton > button {
        width: 100%;
        height: 55px;
        border-radius: 12px;
        border: none;
        font-size: 17px;
        font-weight: 700;
        color: white;
        background: linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6
        );
        transition: 0.25s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🔍 Quora Duplicate Question Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Compare two questions using Natural Language Processing '
    'and Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# QUESTION 1
# =========================================================

st.markdown(
    '<div class="question-card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="question-label">📝 Question 1</div>',
    unsafe_allow_html=True
)

q1 = st.text_area(
    "Question 1",
    placeholder="Enter your first question...",
    height=100,
    label_visibility="collapsed"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# QUESTION 2
# =========================================================

st.markdown(
    '<div class="question-card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="question-label">📝 Question 2</div>',
    unsafe_allow_html=True
)

q2 = st.text_area(
    "Question 2",
    placeholder="Enter your second question...",
    height=100,
    label_visibility="collapsed"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# CHECK BUTTON
# =========================================================

if st.button(
    "🔎 Check for Duplicate",
    use_container_width=True
):

    # Check whether both questions are entered

    if not q1.strip() or not q2.strip():

        st.warning(
            "⚠️ Please enter both questions."
        )

    else:

        # Generate features

        with st.spinner("🔄 Analyzing questions..."):

            query = helper.query_point_creator(
                q1,
                q2
            )

            # Prediction

            result = model.predict(query)[0]


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        if result == 1:

            st.error(
                "🔴 Duplicate Questions"
            )

            st.write(
                "The model predicts that both questions "
                "have the same or very similar meaning."
            )

        else:

            st.success(
                "🟢 Not Duplicate"
            )

            st.write(
                "The model predicts that the two questions "
                "have different meanings."
            )