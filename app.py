import os
import io
import json
import csv
import streamlit as st
from moviepy import *
from dotenv import load_dotenv
from groq import Groq
from podcast.speech_text import audio_to_text
from podcast.embedding import store_embedding
from podcast.question_answer import query_vector_database, transcript_chat_completion
from langchain.docstore.document import Document

# Function to save transcription in different formats
def save_transcription(transcription, format='txt'):
    """
    Save transcription in the specified format
    Supports: txt, json, csv
    """
    if format == 'txt':
        return transcription.encode()
    elif format == 'json':
        return json.dumps({'transcription': transcription}).encode()
    elif format == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Transcription'])
        writer.writerow([transcription])
        return output.getvalue().encode()
    else:
        raise ValueError(f"Unsupported format: {format}")

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Ensure the API_KEY is loaded
if API_KEY is None:
    st.error("API key not found. Please set the GROQ_API_KEY in your .env file.")
    st.stop()

# Initialize the Groq client
client = Groq(api_key=API_KEY)

# Ensure output directories exist
mp3_file_folder = "uploaded_files"
mp3_chunk_folder = "chunks"
os.makedirs(mp3_file_folder, exist_ok=True)
os.makedirs(mp3_chunk_folder, exist_ok=True)

st.title("Podcast Insight Explorer")

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
</style>
""", unsafe_allow_html=True)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Custom JavaScript for drag and drop
st.markdown("""
<script>
const fileUploader = document.querySelector('.stFileUploader');
const dropArea = document.querySelector('.file-uploader');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
  dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
  dropArea.classList.add('highlight');
}

function unhighlight(e) {
  dropArea.classList.remove('highlight');
}
</script>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title fade-in'>Podcast Insight Explorer</h1>", unsafe_allow_html=True)

# Session state initialization
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = []
if "docsearch" not in st.session_state:
    st.session_state.docsearch = None

# File uploader
st.markdown("<div class='card file-uploader fade-in'>", unsafe_allow_html=True)
st.markdown("<h2 class='card-title'>Upload Your Podcast</h2>", unsafe_allow_html=True)
st.markdown("<p class='file-uploader-text'>Drag and drop your MP3 file here or click to browse</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type="mp3")
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Checking if the new file is different from the last processed file
    if uploaded_file.name != st.session_state.last_uploaded_file:
        st.session_state.transcriptions = []
        st.session_state.docsearch = None
        st.session_state.last_uploaded_file = uploaded_file.name

    # Save and process the uploaded file
    filepath = os.path.join(mp3_file_folder, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    audio = AudioFileClip(filepath)
    chunk_length = 60  # seconds

    # Process and transcribe each chunk
    if not st.session_state.transcriptions:
        st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
        st.markdown("<h2 class='card-title'>Transcribing Podcast</h2>", unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, start in enumerate(range(0, int(audio.duration), chunk_length)):
            end = min(start + chunk_length, int(audio.duration))
            audio_chunk = audio.subclipped(start, end)
            chunk_filename = os.path.join(mp3_chunk_folder, f"chunk_{start}.mp3")
            audio_chunk.write_audiofile(chunk_filename, logger=None)

            transcription = audio_to_text(chunk_filename)
            st.session_state.transcriptions.append(transcription)

            # Update progress bar and status
            progress = (i + 1) / ((int(audio.duration) // chunk_length) + 1)
            progress_bar.progress(progress)
            status_text.text(f"Processing chunk {i+1} of {(int(audio.duration) // chunk_length) + 1}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Combine all transcriptions
        combined_transcription = " ".join(st.session_state.transcriptions)
        
        st.markdown("<div class='card transcription-box fade-in'>", unsafe_allow_html=True)
        st.markdown("<h2 class='card-title'>Transcription Preview</h2>", unsafe_allow_html=True)
        st.write(f"{combined_transcription[:500]}...")
        
        # Download buttons for different formats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="Download as TXT",
                data=save_transcription(combined_transcription, 'txt'),
                file_name=f"{uploaded_file.name}_transcription.txt",
                mime="text/plain",
                key='txt_download',
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="Download as JSON",
                data=save_transcription(combined_transcription, 'json'),
                file_name=f"{uploaded_file.name}_transcription.json",
                mime="application/json",
                key='json_download',
                use_container_width=True
            )
        
        with col3:
            st.download_button(
                label="Download as CSV",
                data=save_transcription(combined_transcription, 'csv'),
                file_name=f"{uploaded_file.name}_transcription.csv",
                mime="text/csv",
                key='csv_download',
                use_container_width=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Generate embeddings and store in Pinecone
        documents = [Document(page_content=combined_transcription)]
        st.session_state.docsearch = store_embedding(documents)

    # User query
    st.markdown("<div class='card question-input fade-in'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Ask About Your Podcast</h2>", unsafe_allow_html=True)
    user_question = st.text_input("", placeholder="Type your question here...")
    st.markdown("</div>", unsafe_allow_html=True)

    if user_question and st.session_state.docsearch:
        with st.spinner("🧠 Analyzing podcast content..."):
            relevant_transcripts = query_vector_database(st.session_state.docsearch, user_question)
            response = transcript_chat_completion(client, relevant_transcripts, user_question)
        
        st.markdown("<div class='card response-box fade-in'>", unsafe_allow_html=True)
        st.markdown("<h2 class='card-title'>Podcast Insights</h2>", unsafe_allow_html=True)
        st.write(response)
        st.markdown("</div>", unsafe_allow_html=True)


st.markdown("""
<div class='app-info fade-in'>
    <h3>About Podcast Insight Explorer</h3>
    <p>This AI-powered assistant transcribes and analyzes your podcast episodes, allowing you to extract valuable insights through natural language queries.</p>
    <p>Simply upload an MP3 file, wait for processing, then ask questions about the content to uncover hidden gems in your podcasts!</p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<footer style='text-align: center; margin-top: 50px; padding: 20px; background-color: #f1f5f9; border-radius: 10px;'>
    <p>Powered by Groq AI and Streamlit | Created with ❤️ </p>
</footer>
""", unsafe_allow_html=True)