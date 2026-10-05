import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import io
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SonoVision",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LANGUAGE TEXT
# ============================================================

TEXT = {

    "English": {
        "title": "SonoVision",
        "subtitle": "Ultrasound Image Analysis Using Computer Vision",
        "tagline": "Automatic carotid lumen segmentation and geometric analysis",

        "language": "Language",
        "upload_title": "Upload Ultrasound Image",
        "upload_help": "Upload a JPG, JPEG or PNG ultrasound image",
        "browse": "Choose an ultrasound image",
        "sample": "Try Sample Image",
        "analyze": "Analyze Image",
        "analyzing": "Analyzing ultrasound image...",

        "original": "Original Image",
        "segmented": "Lumen Segmentation",
        "results": "Analysis Results",

        "area": "Area",
        "perimeter": "Perimeter",
        "diameter": "Equivalent Diameter",
        "circularity": "Circularity",
        "bbox": "Bounding Box",

        "summary_title": "Analysis Summary",
        "summary_text":
            "The system automatically identified the largest lumen region "
            "and calculated its geometric properties.",

        "how_title": "How SonoVision Works",
        "step1": "Upload Image",
        "step2": "Preprocessing",
        "step3": "U-Net Segmentation",
        "step4": "Lumen Detection",
        "step5": "Geometric Analysis",
        "step6": "Visualization",

        "notice_title": "Important Notice",
        "notice":
            "SonoVision is an educational and research-oriented image "
            "analysis tool. It automatically segments the carotid lumen "
            "and calculates image-based geometric measurements. "
            "Measurements are reported in pixels and are not clinical "
            "measurements. This system is NOT intended for medical "
            "diagnosis or clinical decision-making.",

        "download": "Download Results",
        "another": "Analyze Another Image",
        "complete": "Analysis Complete",

        "scan1": "Scanning ultrasound image...",
        "scan2": "Preprocessing image...",
        "scan3": "Running lumen segmentation...",
        "scan4": "Detecting lumen boundary...",
        "scan5": "Calculating geometric measurements...",
        "scan6": "Preparing visualization...",

        "area_unit": "px²",
        "pixel_unit": "px",

        "error_model":
            "Model file could not be loaded. Please make sure "
            "carotid_small_unet.pth is in the same folder as app.py.",

        "error_image":
            "Please upload a valid ultrasound image.",

        "no_region":
            "No suitable lumen region was detected in this image.",

        "download_file": "sonovision_analysis.csv"
    },


    "Hindi": {
        "title": "SonoVision",
        "subtitle": "कंप्यूटर विज़न द्वारा अल्ट्रासाउंड इमेज विश्लेषण",
        "tagline": "कैरोटिड ल्यूमेन का स्वचालित सेगमेंटेशन और ज्यामितीय विश्लेषण",

        "language": "भाषा",
        "upload_title": "अल्ट्रासाउंड इमेज अपलोड करें",
        "upload_help": "JPG, JPEG या PNG अल्ट्रासाउंड इमेज अपलोड करें",
        "browse": "अल्ट्रासाउंड इमेज चुनें",
        "sample": "सैंपल इमेज देखें",
        "analyze": "इमेज का विश्लेषण करें",
        "analyzing": "अल्ट्रासाउंड इमेज का विश्लेषण हो रहा है...",

        "original": "मूल इमेज",
        "segmented": "ल्यूमेन सेगमेंटेशन",
        "results": "विश्लेषण के परिणाम",

        "area": "क्षेत्रफल",
        "perimeter": "परिमाप",
        "diameter": "समतुल्य व्यास",
        "circularity": "वृत्ताकारता",
        "bbox": "बाउंडिंग बॉक्स",

        "summary_title": "विश्लेषण सारांश",
        "summary_text":
            "सिस्टम ने स्वचालित रूप से सबसे बड़े ल्यूमेन क्षेत्र "
            "की पहचान की और उसके ज्यामितीय गुणों की गणना की।",

        "how_title": "SonoVision कैसे काम करता है",
        "step1": "इमेज अपलोड",
        "step2": "प्रीप्रोसेसिंग",
        "step3": "U-Net सेगमेंटेशन",
        "step4": "ल्यूमेन पहचान",
        "step5": "ज्यामितीय विश्लेषण",
        "step6": "विज़ुअलाइज़ेशन",

        "notice_title": "महत्वपूर्ण सूचना",
        "notice":
            "SonoVision एक शैक्षणिक और रिसर्च आधारित इमेज विश्लेषण टूल है। "
            "यह कैरोटिड ल्यूमेन का स्वचालित सेगमेंटेशन करता है और "
            "इमेज-आधारित ज्यामितीय माप निकालता है। माप पिक्सेल में दिए जाते हैं "
            "और इन्हें क्लिनिकल माप नहीं माना जाना चाहिए। यह सिस्टम "
            "चिकित्सीय निदान या क्लिनिकल निर्णय लेने के लिए नहीं है।",

        "download": "परिणाम डाउनलोड करें",
        "another": "दूसरी इमेज का विश्लेषण करें",
        "complete": "विश्लेषण पूरा हुआ",

        "scan1": "अल्ट्रासाउंड इमेज स्कैन की जा रही है...",
        "scan2": "इमेज की प्रीप्रोसेसिंग हो रही है...",
        "scan3": "ल्यूमेन सेगमेंटेशन चल रहा है...",
        "scan4": "ल्यूमेन की सीमा पहचानी जा रही है...",
        "scan5": "ज्यामितीय माप निकाले जा रहे हैं...",
        "scan6": "विज़ुअलाइज़ेशन तैयार किया जा रहा है...",

        "area_unit": "px²",
        "pixel_unit": "px",

        "error_model":
            "मॉडल लोड नहीं हो सका। सुनिश्चित करें कि "
            "carotid_small_unet.pth और app.py एक ही फोल्डर में हैं।",

        "error_image":
            "कृपया एक वैध अल्ट्रासाउंड इमेज अपलोड करें।",

        "no_region":
            "इस इमेज में उपयुक्त ल्यूमेन क्षेत्र नहीं मिला।",

        "download_file": "sonovision_analysis.csv"
    },


    "Marathi": {
        "title": "SonoVision",
        "subtitle": "कॉम्प्युटर व्हिजनद्वारे अल्ट्रासाऊंड इमेज विश्लेषण",
        "tagline": "कॅरोटिड ल्यूमेनचे स्वयंचलित सेगमेंटेशन आणि भौमितिक विश्लेषण",

        "language": "भाषा",
        "upload_title": "अल्ट्रासाऊंड इमेज अपलोड करा",
        "upload_help": "JPG, JPEG किंवा PNG अल्ट्रासाऊंड इमेज अपलोड करा",
        "browse": "अल्ट्रासाऊंड इमेज निवडा",
        "sample": "सॅम्पल इमेज वापरा",
        "analyze": "इमेजचे विश्लेषण करा",
        "analyzing": "अल्ट्रासाऊंड इमेजचे विश्लेषण सुरू आहे...",

        "original": "मूळ इमेज",
        "segmented": "ल्यूमेन सेगमेंटेशन",
        "results": "विश्लेषणाचे परिणाम",

        "area": "क्षेत्रफळ",
        "perimeter": "परिमिती",
        "diameter": "समतुल्य व्यास",
        "circularity": "वर्तुळाकारता",
        "bbox": "बाउंडिंग बॉक्स",

        "summary_title": "विश्लेषण सारांश",
        "summary_text":
            "सिस्टमने स्वयंचलितपणे सर्वात मोठ्या ल्यूमेन क्षेत्राची "
            "ओळख करून त्याच्या भौमितिक गुणधर्मांची गणना केली.",

        "how_title": "SonoVision कसे कार्य करते",
        "step1": "इमेज अपलोड",
        "step2": "प्रीप्रोसेसिंग",
        "step3": "U-Net सेगमेंटेशन",
        "step4": "ल्यूमेन ओळख",
        "step5": "भौमितिक विश्लेषण",
        "step6": "व्हिज्युअलायझेशन",

        "notice_title": "महत्त्वाची सूचना",
        "notice":
            "SonoVision हे शैक्षणिक आणि संशोधनासाठी तयार केलेले इमेज "
            "विश्लेषण साधन आहे. हे कॅरोटिड ल्यूमेनचे स्वयंचलित सेगमेंटेशन "
            "करते आणि इमेजवर आधारित भौमितिक मोजमाप देते. मोजमाप पिक्सेलमध्ये "
            "दिलेली आहेत आणि ती क्लिनिकल मोजमाप मानली जाऊ नयेत. हे सिस्टम "
            "वैद्यकीय निदान किंवा क्लिनिकल निर्णयांसाठी वापरण्यासाठी नाही.",

        "download": "परिणाम डाउनलोड करा",
        "another": "दुसऱ्या इमेजचे विश्लेषण करा",
        "complete": "विश्लेषण पूर्ण झाले",

        "scan1": "अल्ट्रासाऊंड इमेज स्कॅन केली जात आहे...",
        "scan2": "इमेजचे प्रीप्रोसेसिंग सुरू आहे...",
        "scan3": "ल्यूमेन सेगमेंटेशन सुरू आहे...",
        "scan4": "ल्यूमेनची सीमा शोधली जात आहे...",
        "scan5": "भौमितिक मोजमाप काढले जात आहेत...",
        "scan6": "व्हिज्युअलायझेशन तयार केले जात आहे...",

        "area_unit": "px²",
        "pixel_unit": "px",

        "error_model":
            "मॉडेल लोड करता आले नाही. carotid_small_unet.pth "
            "आणि app.py एकाच फोल्डरमध्ये आहेत याची खात्री करा.",

        "error_image":
            "कृपया वैध अल्ट्रासाऊंड इमेज अपलोड करा.",

        "no_region":
            "या इमेजमध्ये योग्य ल्यूमेन क्षेत्र सापडले नाही.",

        "download_file": "sonovision_analysis.csv"
    }
}


# ============================================================
# LANGUAGE SELECTOR
# ============================================================

if "language" not in st.session_state:
    st.session_state.language = "English"

language = st.selectbox(
    "🌐 Language",
    ["English", "Hindi", "Marathi"],
    index=["English", "Hindi", "Marathi"].index(
        st.session_state.language
    )
)

st.session_state.language = language
T = TEXT[language]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f6f9fc;
    }

    .main-title {
        font-size: 48px;
        font-weight: 800;
        color: #12344d;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        color: #4c6475;
        margin-top: 0px;
    }

    .tagline {
        font-size: 15px;
        color: #708090;
        margin-bottom: 25px;
    }

    .hero {
        background: linear-gradient(135deg, #ffffff, #eef7fb);
        border: 1px solid #dceaf1;
        border-radius: 22px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 30px rgba(30, 70, 90, 0.06);
    }

    /* HOW SONOVISION WORKS */

[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #c9dce5 !important;
    border-radius: 14px !important;
    margin-top: 10px !important;
    margin-bottom: 20px !important;
}

[data-testid="stExpander"] summary {
    background: #ffffff !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    color: #12344d !important;
    font-weight: 700 !important;
    font-size: 17px !important;
}

[data-testid="stExpander"] summary:hover {
    background: #eef7fb !important;
}

[data-testid="stExpander"] summary p {
    color: #12344d !important;
    font-weight: 700 !important;
}


    .upload-card {
        background: white;
        border: 2px dashed #9bc4d5;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .section-title {
        color: #12344d;
        font-size: 25px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .metric-card {
        background: white;
        border: 1px solid #dce7ed;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        min-height: 115px;
        box-shadow: 0 5px 18px rgba(30, 70, 90, 0.05);
    }

    .metric-label {
        color: #607887;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #12344d;
        font-size: 23px;
        font-weight: 750;
    }

    .notice {
        background: #fff8e8;
        border-left: 6px solid #e5a72e;
        border-radius: 12px;
        padding: 18px;
        margin-top: 25px;
        margin-bottom: 20px;
        color: #4c3b17;
    }

    .summary {
        background: #edf7f5;
        border-left: 6px solid #3a9d8f;
        border-radius: 12px;
        padding: 18px;
        margin-top: 20px;
        color: #244d48;
    }

    .step-card {
        background: white;
        border: 1px solid #dce7ed;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        min-height: 90px;
    }

    .step-number {
        font-size: 20px;
        font-weight: 800;
        color: #237c91;
    }

    .step-text {
        font-size: 14px;
        color: #536875;
        margin-top: 6px;
    }

    .scan-box {
        background: #071923;
        border-radius: 18px;
        padding: 30px;
        color: white;
        text-align: center;
        margin: 20px 0px;
        border: 1px solid #1d5368;
    }

    .scan-line {
        height: 3px;
        background: #54d6e8;
        margin: 20px auto;
        width: 85%;
        animation: scan 1.5s infinite;
        box-shadow: 0 0 14px #54d6e8;
    }

    @keyframes scan {
        0% {
            transform: translateX(-20%);
            opacity: 0.3;
        }

        50% {
            transform: translateX(20%);
            opacity: 1;
        }

        100% {
            transform: translateX(-20%);
            opacity: 0.3;
        }
    }

    .scan-circle {
        width: 90px;
        height: 90px;
        border: 3px solid #54d6e8;
        border-radius: 50%;
        margin: auto;
        animation: pulse 1.5s infinite;
        box-shadow: 0 0 20px rgba(84,214,232,0.4);
    }

    @keyframes pulse {
        0% {
            transform: scale(0.85);
            opacity: 0.6;
        }

        50% {
            transform: scale(1);
            opacity: 1;
        }

        100% {
            transform: scale(0.85);
            opacity: 0.6;
        }
    }

    .complete {
        background: #eaf8f1;
        color: #23734d;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        font-weight: 700;
        margin: 15px 0;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(32, 16)

        self.out = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        b = self.bottleneck(
            self.pool(e3)
        )

        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.out(d1)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = SmallUNet()

    model.load_state_dict(
        torch.load(
            "carotid_small_unet.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


try:

    model = load_model()

except Exception:

    st.error(T["error_model"])
    st.stop()


# ============================================================
# HEADER
# ============================================================
st.markdown(
    f'''<div class="hero"><div class="main-title">🔬 {T["title"]}</div><div class="subtitle">{T["subtitle"]}</div><div class="tagline">{T["tagline"]}</div></div>''',
    unsafe_allow_html=True
)

# ============================================================
# HOW IT WORKS
# ============================================================

with st.expander(f"🔬 {T['how_title']}"):

    cols = st.columns(6)

    steps = [
        T["step1"],
        T["step2"],
        T["step3"],
        T["step4"],
        T["step5"],
        T["step6"]
    ]

    for i, (col, step) in enumerate(zip(cols, steps)):

        with col:

            st.markdown(
    f'<div class="step-card"><div class="step-number">{i + 1}</div><div class="step-text">{step}</div></div>',
    unsafe_allow_html=True
)

# ============================================================
# UPLOAD SECTION
# ============================================================
# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    f'<div class="section-title">📤 {T["upload_title"]}</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    T["upload_help"],
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)


# ============================================================
# SAMPLE IMAGE
# ============================================================


# ============================================================
# SAMPLE IMAGE
# ============================================================



# ============================================================
# IMAGE PROCESSING FUNCTION
# ============================================================

def analyze_image(pil_image):

    original = np.array(
        pil_image.convert("RGB")
    )

    original_h, original_w = original.shape[:2]

    gray = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2GRAY
    )

    resized = cv2.resize(
        gray,
        (128, 128)
    )

    normalized = resized.astype(
        np.float32
    ) / 255.0

    tensor = torch.tensor(
        normalized,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        prediction = model(tensor)

        probability = torch.sigmoid(
            prediction
        )

        mask = (
            probability[0, 0].numpy() > 0.5
        ).astype(np.uint8) * 255

    mask_original = cv2.resize(
        mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    contours, _ = cv2.findContours(
        mask_original,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        largest_contour
    )

    if area <= 0:
        return None

    perimeter = cv2.arcLength(
        largest_contour,
        True
    )

    equivalent_diameter = np.sqrt(
        (4 * area) / np.pi
    )

    if perimeter > 0:

        circularity = (
            4 * np.pi * area
        ) / (
            perimeter ** 2
        )

    else:

        circularity = 0

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    overlay = original.copy()

    # Semi-transparent segmentation overlay
    segmentation_layer = original.copy()

    cv2.drawContours(
        segmentation_layer,
        [largest_contour],
        -1,
        (40, 210, 130),
        -1
    )

    overlay = cv2.addWeighted(
        original,
        0.68,
        segmentation_layer,
        0.32,
        0
    )

    # Draw boundary
    cv2.drawContours(
        overlay,
        [largest_contour],
        -1,
        (0, 255, 255),
        3
    )

    measurements = {
        "Area (px²)": round(area, 2),
        "Perimeter (px)": round(perimeter, 2),
        "Equivalent Diameter (px)": round(
            equivalent_diameter,
            2
        ),
        "Circularity": round(
            circularity,
            4
        ),
        "Bounding Box Width (px)": int(w),
        "Bounding Box Height (px)": int(h)
    }

    return {
        "original": original,
        "mask": mask_original,
        "overlay": overlay,
        "measurements": measurements
    }


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_file is not None:

    try:

        pil_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            pil_image,
            caption=T["original"],
            use_container_width=True
        )

        analyze_button = st.button(
            f"🔬 {T['analyze']}",
            type="primary",
            use_container_width=True
        )

        if analyze_button:

            # Custom loading animation
            placeholder = st.empty()

            scan_messages = [
                T["scan1"],
                T["scan2"],
                T["scan3"],
                T["scan4"],
                T["scan5"],
                T["scan6"]
            ]

            for message in scan_messages:

                placeholder.markdown(
    f'''<div class="scan-box"><div class="scan-circle"></div><div class="scan-line"></div><h3>{message}</h3></div>''',
    unsafe_allow_html=True
)

                time.sleep(0.35)

            result = analyze_image(
                pil_image
            )

            placeholder.empty()

            if result is None:

                st.error(
                    T["no_region"]
                )

                st.stop()

            st.markdown(
                f"""
                <div class="complete">
                    ✓ {T["complete"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Save result in session state
            st.session_state.analysis_result = result


    except Exception:

        st.error(
            T["error_image"]
        )


# ============================================================
# SHOW RESULTS
# ============================================================

if "analysis_result" in st.session_state:

    result = st.session_state.analysis_result

    st.markdown(
        f'<div class="section-title">🔬 {T["results"]}</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            result["original"],
            caption=T["original"],
            use_container_width=True
        )

    with col2:

        st.image(
            result["overlay"],
            caption=T["segmented"],
            use_container_width=True
        )

    # ========================================================
    # MEASUREMENT CARDS
    # ========================================================

    measurements = result["measurements"]

    metric_cols = st.columns(5)

    metric_data = [
        (
            T["area"],
            f'{measurements["Area (px²)"]} {T["area_unit"]}'
        ),

        (
            T["perimeter"],
            f'{measurements["Perimeter (px)"]} {T["pixel_unit"]}'
        ),

        (
            T["diameter"],
            f'{measurements["Equivalent Diameter (px)"]} {T["pixel_unit"]}'
        ),

        (
            T["circularity"],
            f'{measurements["Circularity"]:.3f}'
        ),

        (
            T["bbox"],
            f'{measurements["Bounding Box Width (px)"]} × '
            f'{measurements["Bounding Box Height (px)"]} {T["pixel_unit"]}'
        )
    ]

    for col, (label, value) in zip(
        metric_cols,
        metric_data
    ):

        with col:

            st.markdown(
    f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
    unsafe_allow_html=True
   ) 

    # ========================================================
    # ANALYSIS SUMMARY
    # ========================================================
    st.markdown(
    f'<div class="summary"><h4>📝 {T["summary_title"]}</h4><p>{T["summary_text"]}</p></div>',
    unsafe_allow_html=True
    )

    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    csv_data = pd.DataFrame(
        [measurements]
    ).to_csv(
        index=False
    )

    st.download_button(
        label=f"⬇️ {T['download']}",
        data=csv_data,
        file_name=T["download_file"],
        mime="text/csv",
        use_container_width=True
    )

    # ========================================================
    # ANALYZE ANOTHER
    # ========================================================

    if st.button(
        f"🔄 {T['another']}",
        use_container_width=True
    ):

        del st.session_state.analysis_result

        st.rerun()


# ============================================================
# IMPORTANT NOTICE
# ============================================================

st.markdown(
    f'''<div class="notice"><h4>⚠️ {T["notice_title"]}</h4><p>{T["notice"]}</p></div>''',
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '''<div style="text-align:center; color:#71828c; font-size:13px; padding:25px 0px 10px 0px;">
        SonoVision • Ultrasound Image Analysis Using Computer Vision
    </div>''',
    unsafe_allow_html=True
)
