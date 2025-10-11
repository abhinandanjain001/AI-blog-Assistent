import streamlit as st
import mimetypes
from google import genai
from google.genai import types
from apikey import google_gemini_api_key, openai_api_key, huggingface_token
from huggingface_hub import InferenceClient

# Initialize Gemini client
client = genai.Client(api_key=google_gemini_api_key)

# Streamlit UI setup
st.set_page_config(page_title="BlogCraft - AI Blog Companion", page_icon="📝")
st.title("📝 BlogCraft - Your AI Blog Writing Companion")
st.subheader("Craft the perfect blog post with AI assistance!")

# Sidebar inputs
with st.sidebar:
    st.header("📋 Blog Details")
    blog_title = st.text_input("Blog Title", placeholder="e.g., The Future of AI in Education")
    keywords = st.text_input("Keywords (comma-separated)", placeholder="AI, education, innovation")
    num_words = st.slider("Number of Words", min_value=250, max_value=1000, step=250, value=500)
    num_images = st.number_input("Number of Images", min_value=1, max_value=5, step=1, value=1)
    submit_button = st.button("🚀 Generate Blog")

# Generate blog and images
if submit_button:
    if not blog_title or not keywords:
        st.error("Please enter both a title and keywords.")
    else:
        st.info("⏳ Generating your blog, please wait...")

        # Dynamic blog prompt
        prompt_text = f"""
        Write a comprehensive, engaging blog post titled "{blog_title}".
        Include these keywords: {keywords}.
        The blog should be about {num_words} words long, informative, and conversational.
        Keep it suitable for an online audience.
        """

        # ✅ Generate blog text (Gemini 2.0 Flash)
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt_text
            )
            blog_content = response.text.strip()
            st.success("✅ Blog generated successfully!")
            st.markdown("### 📝 Generated Blog Content")
            st.write(blog_content)
        except Exception as e:
            st.error(f"Blog generation failed: {e}")
            st.stop()

        st.divider()

        # ✅ Generate AI images using Hugging Face (Free)
        st.subheader("🖼️ AI-Generated Images")
        try:
            hf_client = InferenceClient(token=huggingface_token)
        except Exception as e:
            st.error(f"Image setup failed: {e}")
            st.stop()

        for i in range(num_images):
            img_prompt = (
                f"An illustrative, professional, blog-style image for an article titled "
                f"'{blog_title}' about {keywords}. Digital art, clean, high quality, no text."
            )
            try:
                with st.spinner(f"🎨 Generating image {i+1}..."):
                    image = hf_client.text_to_image(
                        prompt=img_prompt,
                        model="stabilityai/stable-diffusion-xl-base-1.0"
                    )
                    st.image(image, caption=f"Generated Image {i+1}", use_column_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not generate image {i+1}: {e}")

        st.divider()

        # ✅ Blog download option
        st.download_button(
            label="📥 Download Blog as Text File",
            data=blog_content,
            file_name=f"{blog_title.replace(' ', '_')}.txt",
            mime="text/plain"
        )
