import streamlit as st
import google.generativeai as genai

# Page config
st.set_page_config(page_title="AI Blog Generator", page_icon="📝", layout="wide")

# Get Gemini API key from Streamlit secrets
def get_gemini_key():
    try:
        api_keys = st.secrets.get("api_keys", {})
        return api_keys.get("google_gemini_api_key")
    except:
        return None


GEMINI_API_KEY = get_gemini_key()

if not GEMINI_API_KEY:
    st.error("Gemini API key not found in Streamlit secrets.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")


# Generate Blog
def generate_blog(prompt):

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Blog generation error: {e}")
        return None


# UI
st.title("📝 AI Blog Generator")
st.subheader("Generate blogs using Google Gemini AI")

# Sidebar
with st.sidebar:

    blog_title = st.text_input(
        "Blog Title",
        "Future of Artificial Intelligence"
    )

    keywords = st.text_input(
        "Keywords",
        "AI, Machine Learning, Technology"
    )

    blog_length = st.slider(
        "Blog Length",
        200,
        1500,
        500
    )

    generate = st.button("Generate Blog")


# Session state
if "blog" not in st.session_state:
    st.session_state.blog = ""


# Generate blog
if generate:

    prompt = f"""
Write a detailed blog titled '{blog_title}' about {keywords}.

Include:
- Introduction
- Key insights
- Examples
- Conclusion

Approximate length: {blog_length} words.
"""

    with st.spinner("Generating blog with Gemini..."):

        blog = generate_blog(prompt)

    if blog:

        st.session_state.blog = blog

        st.success("Blog generated!")

        st.markdown(blog)

    else:
        st.warning("Blog generation failed.")


# Download
if st.session_state.blog:

    st.download_button(
        "Download Blog",
        st.session_state.blog,
        file_name="blog.txt"
    )