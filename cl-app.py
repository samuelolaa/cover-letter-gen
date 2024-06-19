import os
import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)
import pyperclip

# Initialize Vertex AI
PROJECT_ID = os.getenv("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.getenv("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)

@st.cache_resource
def load_models():
    """Load the generative models for text generation."""
    text_model_pro = GenerativeModel("gemini-1.0-pro")
    return text_model_pro

def get_gemini_pro_text_response(model: GenerativeModel, prompt: str):
    """Generate a response using the Gemini model."""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    generation_config = GenerationConfig(
        temperature=0.7,
        max_output_tokens=1024,
    )

    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    final_response = []
    for response in responses:
        try:
            final_response.append(response.text)
        except IndexError:
            final_response.append("")
            continue
    return " ".join(final_response)

def main():
    st.title("Personalized Cover Letter Generator")

    # Initialize session state for storing previous cover letters
    if 'cover_letters' not in st.session_state:
        st.session_state['cover_letters'] = []

    # Tabs for the interface
    tab1, tab2 = st.tabs(["Generate Cover Letter", "Previous Cover Letters"])

    with tab1:
        st.header("Enter the Job Listing and Description")
        job_listing = st.text_input("Job Listing")
        job_description = st.text_area("Job Description", height=200)

        st.header("Enter Your Work Experience and Skills")
        experience_skills = st.text_area("Work Experience and Skills", height=300)

        if st.button("Generate Cover Letter"):
            # Construct the prompt
            prompt = f"""
            Generate a cover letter using the following details:
            Job Listing: {job_listing}
            Job Description: {job_description}
            Work Experience and Skills: {experience_skills}
            """

            text_model_pro = load_models()
            config = GenerationConfig(
                temperature=0.8,
                max_output_tokens=2048,
            )

            with st.spinner("Generating your cover letter using Gemini 1.0 Pro ..."):
                cover_letter = get_gemini_pro_text_response(
                    text_model_pro,
                    prompt
                )
                st.subheader("Generated Cover Letter")
                st.write(cover_letter)

                # Store the generated cover letter in session state
                st.session_state.cover_letters.append(cover_letter)

                # Add a copy button
                if st.button("Copy Cover Letter"):
                    pyperclip.copy(cover_letter)
                    st.success("Cover letter copied to clipboard!")

    with tab2:
        st.header("Previous Cover Letters")
        if st.session_state.cover_letters:
            for i, letter in enumerate(st.session_state.cover_letters):
                with st.expander(f"Cover Letter {i+1}"):
                    st.write(letter)
                    if st.button(f"Copy Cover Letter {i+1}", key=f"copy_{i}"):
                        pyperclip.copy(letter)
                        st.success(f"Cover Letter {i+1} copied to clipboard!")

if __name__ == "__main__":
    main()
