import streamlit as st
from huggingface_hub import InferenceClient
import os

st.set_page_config(page_title="AI Blog Writer", page_icon="📝", layout="wide")

# Get HF Token
def get_huggingface_token():
    try:
        api_keys = st.secrets.get("api_keys", {})
        return api_keys.get("huggingface_token")
    except:
        return os.getenv("HUGGINGFACE_TOKEN")

HUGGINGFACE_TOKEN = get_huggingface_token()

if not HUGGINGFACE_TOKEN:
    st.error("HuggingFace token missing")
    st.stop()


# TEXT CLIENT
@st.cache_resource
def get_text_client():
    return InferenceClient(
        model="google/flan-t5-large",
        provider="hf-inference",
        token=HUGGINGFACE_TOKEN
    )


# IMAGE CLIENT
@st.cache_resource
def get_image_client():
    return InferenceClient(
        model="stabilityai/stable-diffusion-2",
        provider="hf-inference",
        token=HUGGINGFACE_TOKEN
    )


# Generate Blog
def generate_blog(prompt):

    try:
        client = get_text_client()

        response = client.text_generation(
            prompt,
            max_new_tokens=400
        )

        return response

    except Exception as e:
        st.error(f"Text generation error: {e}")
        return None


# Generate Image
def generate_image(prompt):

    try:
        client = get_image_client()

        image = client.text_to_image(prompt)

        return image

    except Exception as e:
        st.error(f"Image generation error: {e}")
        return None


# UI
st.title("📝 AI Blog Generator")

with st.sidebar:

    title = st.text_input("Blog Title", "Future of Artificial Intelligence")

    keywords = st.text_input(
        "Keywords",
        "AI, Machine Learning, Technology"
    )

    images = st.slider("Images", 1, 3, 1)

    generate = st.button("Generate Blog")


if "blog" not in st.session_state:
    st.session_state.blog = ""


if generate:

    prompt = f"""
Write a detailed blog titled '{title}' about {keywords}.
Include introduction, main points and conclusion.
"""

    with st.spinner("Generating blog..."):

        blog = generate_blog(prompt)

    if blog:

        st.session_state.blog = blog

        st.success("Blog Generated")

        st.write(blog)

    else:
        st.warning("Blog generation failed")


if st.session_state.blog:

    st.subheader("Generated Images")

    for i in range(images):

        img_prompt = f"Illustration for blog about {keywords}"

        with st.spinner("Generating image..."):

            img = generate_image(img_prompt)

        if img:
            st.image(img)


if st.session_state.blog:

    st.download_button(
        "Download Blog",
        st.session_state.blog,
        file_name="blog.txt"
    )