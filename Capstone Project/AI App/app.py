import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import warnings

# Stop the MedCLIP library from crashing when GPU not detected
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    device = "cpu"
else:
    device = "cuda"

# Silence noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPProcessor

# ---- 1. Page Configuration ----
st.set_page_config(
    page_title="MedCLIP Clinical Diagnostic Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to fix the visibility of the metric score
st.markdown("""
    <style>
    /* The main metric card */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa !important;
        padding: 20px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* FORCE LABEL VISIBILITY (Model Confidence Score) */
    div[data-testid="stMetricLabel"] > div {
        color: #333333 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    
    /* FORCE VALUE VISIBILITY (The Percentage) */
    div[data-testid="stMetricValue"] > div {
        color: #000000 !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ---- 2. Load Model & Processor ----
@st.cache_resource
def load_medclip_model():
    processor = MedCLIPProcessor()
    model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
    
    try:
        # Load weights on the detected device (CPU or CUDA)
        state_dict = torch.load("MedCLIP_LoRA_best_weights.pth", map_location=device)
        model.load_state_dict(state_dict, strict=False)
        load_status = f"✅ MedCLIP-LoRA Active ({device.upper()})"
    except FileNotFoundError:
        load_status = "⚠️ LoRA weights not found. Using Base MedCLIP Model."
    
    model.to(device)
    model.eval()
    return model, processor, load_status

model, processor, status = load_medclip_model()

# ---- 3. Sidebar ----
st.sidebar.header("Patient Information")
patient_id = st.sidebar.text_input("Patient ID", value="PX-4502")
study_date = st.sidebar.date_input("Study Date")
st.sidebar.markdown("---")
st.sidebar.info(status)

# ---- 4. Main UI ----
st.title("MedCLIP Diagnostic Assistant")
st.write("Diagnostic support system for automated chest radiography screening.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Step 1: Upload X-ray")
    uploaded_file = st.file_uploader("Select Chest X-ray Scan (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption=f"Patient Scan: {uploaded_file.name}", width='stretch')
    else:
        st.info("Awaiting image upload to begin clinical analysis.")

with col2:
    st.subheader("Step 2: AI Interpretation")
    if uploaded_file is not None:
        # Clinical queries for classification
        cls_prompts = ["chest x-ray with no findings", "chest x-ray with pathology"]
        
        if st.button("Run Clinical Inference"):
            with st.spinner("Analyzing image features..."):
                # Preprocessing
                inputs = processor(
                    text=cls_prompts, 
                    images=input_image, 
                    return_tensors="pt", 
                    padding=True
                ).to(device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = None
                    
                    # 1. Attempt to access as attributes (Object style)
                    for attr in ['logits_per_image', 'logits']:
                        if hasattr(outputs, attr):
                            logits = getattr(outputs, attr)
                            break
                    
                    # 2. Attempt to access as dictionary keys (Dict style)
                    if logits is None and isinstance(outputs, dict):
                        for key in ['logits_per_image', 'logits']:
                            if key in outputs:
                                logits = outputs[key]
                                break
                    
                    # 3. Final fallback: Grab first dict value if naming is unknown
                    if logits is None and isinstance(outputs, dict) and outputs:
                        logits = list(outputs.values())[0]

                    if logits is not None:
                        # Process resulting probabilities
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                        
                        st.markdown("### Findings")
                        # Index 0: Normal, Index 1: Pathology
                        has_pathology = probs[1] > probs[0]
                        prediction = "Pathology Detected" if has_pathology else "Normal Findings"
                        confidence = probs[1] if has_pathology else probs[0]
                        
                        if has_pathology:
                            st.error(f"**Diagnostic Status:** {prediction}")
                        else:
                            st.success(f"**Diagnostic Status:** {prediction}")
                        
                        st.metric("Model Confidence Score", f"{confidence*100:.2f}%")
                    else:
                        st.error("Unexpected model output format.")
                        st.write("Debug - Output Type:", type(outputs))
                        if isinstance(outputs, dict):
                            st.write("Debug - Available Keys:", list(outputs.keys()))
                
                st.divider()
                st.write("**Clinician Verification**")
                feedback = st.radio("Do you agree with this AI assessment?", ["Pending Review", "Agree", "Disagree"], horizontal=True)
                if feedback != "Pending Review":
                    st.toast("Feedback logged for model monitoring.")
    else:
        st.write("Upload an image in Step 1 to generate findings.")