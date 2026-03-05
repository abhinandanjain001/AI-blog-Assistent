import streamlit as st
import requests
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv


# --- PAGE CONFIG MUST BE FIRST ---
st.set_page_config(page_title="HuggingFace BlogWriter", page_icon="📝", layout="wide")

# Load environment variables
load_dotenv()

# --- Hugging Face API Configuration ---
def get_huggingface_token():
    """Resolve HF token from Streamlit secrets (nested/flat) or env vars."""
    try:
        api_keys = st.secrets.get("api_keys", {})
        token = api_keys.get("huggingface_token")
        if token:
            return token
    except Exception:
        pass

    try:
        token = st.secrets.get("huggingface_token") or st.secrets.get("HUGGINGFACE_TOKEN")
        if token:
            return token
    except Exception:
        pass

    return os.getenv("HUGGINGFACE_TOKEN") or os.getenv("huggingface_token")


HUGGINGFACE_TOKEN = get_huggingface_token()

if not HUGGINGFACE_TOKEN:
    st.error("HUGGINGFACE_TOKEN not found in Streamlit secrets or .env file. Please configure your API keys.")
    st.stop()

# Initialize clients lazily when needed
@st.cache_resource
def get_text_client():
    client_kwargs = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "token": HUGGINGFACE_TOKEN,
    }
    try:
        return InferenceClient(**client_kwargs, base_url="https://router.huggingface.co")
    except TypeError:
        # Older huggingface_hub versions do not support base_url.
        return InferenceClient(**client_kwargs)

@st.cache_resource
def get_image_client():
    client_kwargs = {
        "model": "stabilityai/stable-diffusion-3.5-large-turbo",
        "token": HUGGINGFACE_TOKEN,
    }
    try:
        return InferenceClient(**client_kwargs, base_url="https://router.huggingface.co")
    except TypeError:
        # Older huggingface_hub versions do not support base_url.
        return InferenceClient(**client_kwargs)

# --- Functions for Hugging Face Inference ---
@st.cache_data(show_spinner=False)
def generate_blog_content(prompt, max_length=500, temperature=0.9):
    """
    Generates text using Hugging Face InferenceClient with a reliable model.
    """
    try:
        text_client = get_text_client()
        response = text_client.text_generation(
            prompt=prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True
        )
        
        if response:
            return response.strip()
        else:
            st.error("Text generation returned empty response")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network or API error during text generation: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during text generation: {e}")
        return None


@st.cache_data(show_spinner=False)
def generate_hf_image(prompt_text):
    """Generates an image using Hugging Face InferenceClient."""
    try:
        # The image_client's text_to_image method handles the API call
        image_client = get_image_client()
        image = image_client.text_to_image(
            prompt=prompt_text
        )
        return image
    except Exception as e:
        st.error(f"Image generation failed: {e}")
        return None

# --- Streamlit UI ---
st.title("📝 Hugging Face BlogWriter")
st.subheader("Generate blogs & images using only HF Inference APIs!")

# --- Sidebar ---
with st.sidebar:
    st.header("📋 Blog Settings")
    blog_title = st.text_input("Blog Title", "The Future of Open Source AI")
    keywords = st.text_input("Keywords (comma-separated)", "Open Source AI, LLMs, community, development")
    blog_length = st.slider("Blog Length (approx. words)", 100, 700, 300) # GPT-2 max_length is in tokens
    num_images = st.slider("Number of Images", 1, 3, 1)
    
    st.subheader("⚙️ Generation Parameters (GPT-2)")
    # GPT-2 is very sensitive to temperature; higher values make it more creative but less coherent.
    gpt2_temperature = st.slider("GPT-2 Temperature", 0.1, 1.5, 0.9, 0.1)
    
    generate_button = st.button("🚀 Generate Blog & Images")

# Initialize session state for blog content
if 'hf_blog_content' not in st.session_state:
    st.session_state.hf_blog_content = ""

if generate_button:
    if not blog_title or not keywords:
        st.error("Please enter a blog title and keywords.")
    else:
        # --- Blog Generation ---
        st.subheader("✍️ Your Generated Blog")
        blog_prompt = (
            f"Write a comprehensive blog post titled '{blog_title}' focusing on '{keywords}'. "
            f"Start with an engaging introduction, provide several main points with details, "
            f"and conclude with a summary. The blog should be well-structured and approximately {blog_length} words long."
            f"\n\nTitle: {blog_title}\n\n"
        )
        
        with st.spinner("Generating blog content with Mistral... this might take a moment."):
            st.session_state.hf_blog_content = generate_blog_content(
                blog_prompt, 
                max_length=blog_length * 2,
                temperature=gpt2_temperature
            )
            
        if st.session_state.hf_blog_content:
            st.success("Blog content generated!")
            st.markdown(st.session_state.hf_blog_content)
        else:
            st.warning("Could not generate blog content.")

    # --- Image Generation ---
    if st.session_state.hf_blog_content: # Only generate images if blog content exists
        st.subheader("🖼️ Generated Images")
        image_gen_prompts = []
        for i in range(num_images):
            # Create a more specific prompt for each image if desired, or keep general
            image_prompt = (
                f"Professional, digital art illustration for a blog post titled '{blog_title}' "
                f"about {keywords}. Focus on a key concept from the blog. Clean, modern, vibrant style, no text, no abstract shapes."
            )
            image_gen_prompts.append(image_prompt)

        generated_images = []
        for i, img_prompt in enumerate(image_gen_prompts):
            with st.spinner(f"Creating image {i+1}/{num_images} for '{blog_title}'..."):
                img = generate_hf_image(img_prompt)
                if img:
                    generated_images.append((img, f"Image {i+1}: {img_prompt}"))
                    st.image(img, caption=f"Image {i+1} for '{blog_title}'")
                else:
                    st.warning(f"Failed to generate image {i+1}.")
    
    # --- Download Button (only if blog was generated) ---
    if st.session_state.hf_blog_content:
        st.download_button(
            label="📥 Download Blog (TXT)",
            data=st.session_state.hf_blog_content,
            file_name=f"{blog_title.replace(' ', '_').lower()}.txt",
            mime="text/plain"
        )
