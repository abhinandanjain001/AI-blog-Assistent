import streamlit as st
import requests
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Hugging Face API Configuration ---
# Get your Hugging Face API token from .env
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HUGGINGFACE_TOKEN:
    st.error("HUGGINGFACE_TOKEN not found in .env file. Please create a .env file and add your token.")
    st.stop()

# Initialize InferenceClient for text generation (using requests for more control over GPT-2 specifics)
# GPT-2 text generation model
GPT2_API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

# Initialize InferenceClient for image generation
# Stable Diffusion XL for high-quality images
image_client = InferenceClient(token=HUGGINGFACE_TOKEN)
IMAGE_GEN_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


# --- Functions for Hugging Face Inference ---
@st.cache_data(show_spinner=False)
def generate_gpt2_blog_content(prompt, max_length=500, temperature=0.9):
    """
    Generates text using the Hugging Face Inference API for GPT-2.
    Using requests for more fine-grained control over generation parameters.
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_length,
            "temperature": temperature,
            "do_sample": True, # Enables sampling
            "num_return_sequences": 1,
            "eos_token_id": 50256 # End of sequence token for GPT-2
        }
    }
    
    try:
        response = requests.post(GPT2_API_URL, headers=headers, json=payload, timeout=60) # Increased timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        output = response.json()
        
        if output and isinstance(output, list) and 'generated_text' in output[0]:
            # GPT-2 typically returns the input prompt prepended to the generated text.
            # We want to remove the input prompt to get only the new content,
            # but ensure we don't accidentally cut off the start of a sentence.
            generated_text_with_prompt = output[0]['generated_text']
            
            # Find where the prompt ends and the new text begins (approximate)
            # This is a heuristic, GPT-2 might rephrase the beginning slightly
            if generated_text_with_prompt.startswith(prompt):
                return generated_text_with_prompt[len(prompt):].strip()
            return generated_text_with_prompt.strip() # Fallback

        else:
            st.error(f"GPT-2 generation failed: {output}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network or API error during GPT-2 generation: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during GPT-2 generation: {e}")
        return None


@st.cache_data(show_spinner=False)
def generate_hf_image(prompt_text):
    """Generates an image using Hugging Face InferenceClient."""
    try:
        # The image_client's text_to_image method handles the API call
        image = image_client.text_to_image(
            prompt=prompt_text,
            model=IMAGE_GEN_MODEL
        )
        return image
    except Exception as e:
        st.error(f"Image generation failed: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="HuggingFace BlogWriter", page_icon="📝")
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
        
        with st.spinner("Generating blog content with GPT-2... this might take a moment."):
            st.session_state.hf_blog_content = generate_gpt2_blog_content(
                blog_prompt, 
                max_length=blog_length * 2, # GPT-2 max_new_tokens is tokens, not words. ~2x words for tokens.
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
