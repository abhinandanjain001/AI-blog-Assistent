import streamlit as st
import google.generativeai as genai

# =============================
#  Secure API Configuration
# =============================
api_key = st.secrets["api_keys"]["google_gemini"]
genai.configure(api_key=api_key)

# =============================
#  Page Configx
# =============================
st.set_page_config(page_title="🧠 AI Blog Companion", layout="wide")
st.title("🧠 AI Blog Companion")
st.caption("Your AI-powered assistant for writing, researching & creating visuals 🚀")

# =============================
#  Sidebar Design
# =============================
st.sidebar.title("⚙️ Settings")

tone = st.sidebar.selectbox("Blog Tone", ["Professional", "Casual", "Friendly", "Technical"])
length = st.sidebar.slider("Blog Length (words)", 300, 2000, 800, step=100)
include_image = st.sidebar.checkbox("Generate Blog Cover Image", True)
st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Use detailed descriptions for better results.")

# =============================
#  User Input
# =============================
blog_title = st.text_input("📝 Blog Title", placeholder="e.g. The Future of AI in Healthcare")
blog_description = st.text_area("📄 Blog Topic / Description", placeholder="Describe the blog content or key points...")

# =============================
#  Generate Blog
# =============================
if st.button("🚀 Generate Blog"):
    if not blog_title or not blog_description:
        st.warning("⚠️ Please enter both the blog title and description.")
    else:
        with st.spinner("Generating your blog with Gemini... ⏳"):
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                f"Write a detailed blog post titled '{blog_title}' about {blog_description}. "
                f"Tone: {tone}. Length: around {length} words."
            )
            response = model.generate_content(prompt)
            st.subheader("🧾 Generated Blog Post")
            st.write(response.text)

        # =============================
        #  Generate Blog Image (Optional)
        # =============================
        if include_image:
            with st.spinner("Generating cover image... 🎨"):
                image_prompt = f"An artistic blog cover image for: {blog_title}"
                image_model = genai.GenerativeModel("imagen-3.0-generate")  # DALL·E-like model
                image_response = image_model.generate_content(image_prompt)
                st.image(image_response.images[0], caption="🖼️ AI-Generated Cover Image", use_container_width=True)

st.markdown("---")
st.markdown("💬 *Made by Abhinandan Jain | Powered by Google Gemini + Streamlit*")
