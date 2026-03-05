import streamlit as st
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# --- PAGE CONFIG ---
st.set_page_config(page_title="HuggingFace BlogWriter", page_icon="📝", layout="wide")

load_dotenv()

# --- GET HF TOKEN ---
def get_huggingface_token():
    try:
        api_keys = st.secrets.get("api_keys", {})
        token = api_keys.get("huggingface_token")
        if token:
            return token
    except:
        pass

    try:
        token = st.secrets.get("huggingface_token")
        if token:
            return token
    except:
        pass

    return os.getenv("HUGGINGFACE_TOKEN")

HUGGINGFACE_TOKEN = get_huggingface_token()

if not HUGGINGFACE_TOKEN:
    st.error("HUGGINGFACE_TOKEN not found in secrets or env")
    st.stop()

# --- TEXT CLIENT ---
@st.cache_resource
def get_image_client():
    return InferenceClient(
        model="stabilityai/stable-diffusion-xl-base-1.0",
        provider="hf-inference",
        token=HUGGINGFACE_TOKEN
    )

# --- IMAGE CLIENT ---
@st.cache_resource
def get_image_client():
    return InferenceClient(
        model="stabilityai/stable-diffusion-xl-base-1.0",
        token=HUGGINGFACE_TOKEN
    )

# --- TEXT GENERATION ---
def generate_blog_content(prompt, max_length=500, temperature=0.9):

    try:
        client = get_text_client()

        response = client.text_generation(
            prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True
        )

        if response:
            return response.strip()
        return None

    except Exception as e:
        st.error(f"Text generation error: {str(e)}")
        return None


# --- IMAGE GENERATION ---
def generate_hf_image(prompt_text):

    try:
        image_client = get_image_client()

        image = image_client.text_to_image(
            prompt=prompt_text
        )

        return image

    except Exception as e:
        st.error(f"Image generation failed: {e}")
        return None


# --- UI ---
st.title("📝 Hugging Face BlogWriter")
st.subheader("Generate blogs & images using HuggingFace AI models!")

# --- SIDEBAR ---
with st.sidebar:

    st.header("📋 Blog Settings")

    blog_title = st.text_input(
        "Blog Title",
        "The Future of Open Source AI"
    )

    keywords = st.text_input(
        "Keywords",
        "Open Source AI, LLMs, community"
    )

    blog_length = st.slider(
        "Blog Length",
        100,
        700,
        300
    )

    num_images = st.slider(
        "Number of Images",
        1,
        3,
        1
    )

    temperature = st.slider(
        "Creativity (Temperature)",
        0.1,
        1.5,
        0.9
    )

    generate_button = st.button("🚀 Generate Blog")

# --- SESSION STATE ---
if "blog_content" not in st.session_state:
    st.session_state.blog_content = ""

# --- GENERATE BLOG ---
if generate_button:

    if not blog_title or not keywords:
        st.error("Enter blog title and keywords")
    else:

        st.subheader("✍️ Generated Blog")

        prompt = f"""
Write a detailed blog titled '{blog_title}' about {keywords}.

Include:
- Introduction
- Key insights
- Examples
- Conclusion

Blog length approx {blog_length} words.
"""

        with st.spinner("Generating blog using AI..."):

            st.session_state.blog_content = generate_blog_content(
                prompt,
                max_length=blog_length * 2,
                temperature=temperature
            )

        if st.session_state.blog_content:

            st.success("Blog generated!")

            st.markdown(st.session_state.blog_content)

        else:
            st.warning("Could not generate blog content")

# --- IMAGE GENERATION ---
if st.session_state.blog_content:

    st.subheader("🖼️ Generated Images")

    for i in range(num_images):

        img_prompt = f"""
Professional blog illustration about {keywords}.
Modern digital art, clean style.
"""

        with st.spinner(f"Generating image {i+1}"):

            img = generate_hf_image(img_prompt)

            if img:
                st.image(img, caption=f"Image {i+1}")

# --- DOWNLOAD ---
if st.session_state.blog_content:

    st.download_button(
        "📥 Download Blog",
        st.session_state.blog_content,
        file_name=f"{blog_title.replace(' ','_')}.txt"
    )