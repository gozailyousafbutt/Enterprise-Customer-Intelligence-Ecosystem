from io import BytesIO
from pathlib import Path
import sys
from PIL import Image
import streamlit as st

# Dynamic Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_PATH = BASE_DIR / "src"
DATA_PATH = BASE_DIR / "data" / "customer_support_tickets.csv"

if str(SRC_PATH) not in sys.path:
  sys.path.append(str(SRC_PATH))

from utilis import FinalMultimodalRAGPipeline

# Page Config
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("Multimodal Customer Support and Ticket RAG Assistant")
st.write(
    "This app is based on a Multimodal RAG system that processes both images"
    " and text."
)

# Sidebar
with st.sidebar:
  st.header("Project Info")
  st.write("Project: Multimodal RAG-Assistant")
  st.markdown("---")
  st.info("Powered by EasyOCR & LangChain.")


# Load Pipeline with Caching
@st.cache_resource
def load_rag_pipeline():
  try:
    if not DATA_PATH.exists():
      return None, f"Knowledge base dataset not found at: {DATA_PATH}"
    pipeline = FinalMultimodalRAGPipeline(
        csv_path=str(DATA_PATH),
        text_column="Ticket Description",
    )
    return pipeline, None
  except Exception as e:
    return None, str(e)


# Load pipeline and show progress
with st.spinner(
    "Loading AI Models & Knowledge Base... (This may take a minute)"
):
  rag_pipeline, pipeline_error = load_rag_pipeline()

if pipeline_error:
  st.error(f"Pipeline Error: {pipeline_error}")

# UI Elements
uploaded_image_file = st.file_uploader(
    "Upload Customer Support Image/Ticket", type=["png", "jpg", "jpeg"]
)

user_question = st.text_input(
    "Enter your question regarding the image/ticket:",
    value="What issue is reported in the customer support ticket?",
)

if st.button("Generate AI Answer"):
  if rag_pipeline is None:
    st.error("Pipeline is not loaded. Please check the error above.")
  elif uploaded_image_file is not None:
    try:
      image_bytes = uploaded_image_file.getvalue()
      pil_image = Image.open(BytesIO(image_bytes))

      # Display image in Streamlit from memory
      st.image(
          pil_image,
          caption=f"Uploaded Image: {uploaded_image_file.name}",
          use_container_width=True,
      )

      with st.spinner("Processing image text and searching database..."):
        final_ai_response = rag_pipeline.run(
            user_question=user_question, image_path=pil_image
        )

        st.success("Answer Generated Successfully!")
        st.subheader("AI Response:")
        st.write(final_ai_response)

    except Exception as error_message:
      st.error(f"Execution Error: {error_message}")
  else:
    st.warning("Please upload an image first!")