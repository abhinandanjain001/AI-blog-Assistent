import json

import streamlit as st
import streamlit.components.v1 as components
from google import genai


# Page setup
st.set_page_config(page_title="AI Blog + Image Generator", page_icon="📝", layout="wide")


def get_gemini_api_key():
    """Read Gemini key safely from Streamlit secrets."""
    try:
        api_keys = st.secrets.get("api_keys", {})
        return api_keys.get("google_gemini_api_key")
    except Exception:
        return None


def get_gemini_client():
    """Create Gemini client if key exists."""
    key = get_gemini_api_key()
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None


def generate_blog(prompt):
    """Generate blog text with Gemini."""
    client = get_gemini_client()
    if not client:
        st.error("Gemini API key not found in Streamlit secrets.")
        return None

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        st.error(f"Blog generation error: {e}")
        return None


def render_puter_images(prompt_text):
    """Render Puter.js txt2img gallery for multiple models."""
    prompt_json = json.dumps(prompt_text)
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2>Different Models Comparison</h2>
        <div id="images"></div>

        <script src="https://js.puter.com/v2/"></script>
        <script>
            const prompt = {prompt_json};
            const container = document.getElementById('images');
            container.innerHTML = '';

            function renderCard(title, img) {{
                const div = document.createElement('div');
                div.style.marginBottom = '20px';
                div.innerHTML = `<h3>${{title}}</h3>`;
                div.appendChild(img);
                container.appendChild(div);
            }}

            function renderError(title, err) {{
                const div = document.createElement('div');
                div.style.marginBottom = '20px';
                div.innerHTML = `<h3>${{title}}</h3><p style="color:#b00020;">${{String(err)}}</p>`;
                container.appendChild(div);
            }}

            puter.ai.txt2img(prompt, {{ model: "gpt-image-1", quality: "low" }})
                .then(img => renderCard("GPT Image (Quality: low)", img))
                .catch(err => renderError("GPT Image (Quality: low)", err));

            puter.ai.txt2img(prompt, {{ model: "dall-e-3", quality: "hd" }})
                .then(img => renderCard("DALL-E 3 (Quality: hd)", img))
                .catch(err => renderError("DALL-E 3 (Quality: hd)", err));

            puter.ai.txt2img(prompt, {{ model: "gemini-2.5-flash-image-preview" }})
                .then(img => renderCard("Gemini 2.5 Flash Image Preview", img))
                .catch(err => renderError("Gemini 2.5 Flash Image Preview", err));

            puter.ai.txt2img(prompt, {{ model: "stabilityai/stable-diffusion-3-medium" }})
                .then(img => renderCard("Stable Diffusion 3", img))
                .catch(err => renderError("Stable Diffusion 3", err));

            puter.ai.txt2img(prompt, {{ model: "black-forest-labs/FLUX.1-schnell" }})
                .then(img => renderCard("Flux.1 Schnell", img))
                .catch(err => renderError("Flux.1 Schnell", err));
        </script>
    </body>
    </html>
    """
    components.html(html, height=1800, scrolling=True)


# UI
st.title("📝 AI Blog + Image Generator")
st.subheader("Generate blogs with Gemini and images with Puter multi-model comparison")

# Sidebar
with st.sidebar:
    blog_title = st.text_input("Blog Title", "Future of Artificial Intelligence")
    keywords = st.text_input("Keywords", "AI, Machine Learning, Technology")
    blog_length = st.slider("Blog Length", 200, 1500, 500)
    generate_blog_button = st.button("Generate Blog")

st.divider()

st.subheader("🖼️ AI Image Generation (Puter)")
image_prompt = st.text_input(
    "Image Prompt",
    "A futuristic city with flying cars",
)
generate_images_button = st.button("Generate Images (All Models)")


# Session state
if "blog" not in st.session_state:
    st.session_state.blog = ""


# Generate blog
if generate_blog_button:
    prompt = f"""
Write a detailed blog titled '{blog_title}' about {keywords}.

Include:
- Introduction
- Key insights
- Examples
- Conclusion

Approximate length: {blog_length} words.
"""
    with st.spinner("Generating blog..."):
        blog = generate_blog(prompt)

    if blog:
        st.session_state.blog = blog
        st.success("Blog generated!")
        st.markdown(blog)
    else:
        st.warning("Blog generation failed.")


# Download button
if st.session_state.blog:
    st.download_button(
        "Download Blog",
        st.session_state.blog,
        file_name="blog.txt",
    )


# Generate images with Puter
if generate_images_button:
    st.info("If prompted, sign in/authorize Puter in the embedded frame.")
    render_puter_images(image_prompt)