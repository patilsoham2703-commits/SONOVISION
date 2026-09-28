app_code = r'''
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image


# =========================================================
# SonoVision - Exact Training Model Architecture
# =========================================================

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

        # Encoder
        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)

        # Decoder
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

        # Output
        self.out = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )


    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e3)
        )

        # Decoder
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


# =========================================================
# Load Model
# =========================================================

@st.cache_resource
def load_model():

    model = SmallUNet()

    state_dict = torch.load(
        "carotid_small_unet.pth",
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


model = load_model()


# =========================================================
# Image Analysis
# =========================================================

def analyze_image(image):

    original = np.array(
        image.convert("RGB")
    )

    gray = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2GRAY
    )

    # Same input size used during training
    resized = cv2.resize(
        gray,
        (128, 128)
    )

    input_image = (
        resized.astype(np.float32) / 255.0
    )

    tensor = torch.tensor(
        input_image,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # Prediction
    with torch.no_grad():

        prediction = torch.sigmoid(
            model(tensor)
        )

    mask = (
        prediction.squeeze().numpy() > 0.5
    ).astype(np.uint8)

    # Resize mask back to original image size
    mask = cv2.resize(
        mask,
        (
            original.shape[1],
            original.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        return original, mask, None

    # Largest detected region
    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter > 0:

        circularity = (
            4 * np.pi * area
            / (perimeter ** 2)
        )

    else:

        circularity = 0

    if area > 0:

        equivalent_diameter = np.sqrt(
            4 * area / np.pi
        )

    else:

        equivalent_diameter = 0

    x, y, width, height = cv2.boundingRect(
        contour
    )

    # =====================================================
    # Create Visualization
    # =====================================================

    overlay = original.copy()

    overlay[mask > 0] = (
        255,
        0,
        0
    )

    result = cv2.addWeighted(
        original,
        0.70,
        overlay,
        0.30,
        0
    )

    # Draw contour
    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    measurements = {

        "Area": area,

        "Perimeter": perimeter,

        "Equivalent Diameter":
            equivalent_diameter,

        "Circularity":
            circularity,

        "Bounding Box Width":
            width,

        "Bounding Box Height":
            height
    }

    return result, mask, measurements


# =========================================================
# Streamlit Interface
# =========================================================

st.set_page_config(
    page_title="SonoVision",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 SonoVision")

st.subheader(
    "Ultrasound Image Analysis Using Computer Vision"
)

st.write(
    "SonoVision automatically segments the "
    "lumen region in carotid ultrasound images "
    "and provides basic geometric measurements."
)

st.info(
    "Research and visualization project only. "
    "This system is not intended for medical diagnosis."
)


# =========================================================
# Upload Image
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an ultrasound image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.subheader(
        "Uploaded Ultrasound Image"
    )

    st.image(
        image,
        use_container_width=True
    )

    if st.button(
        "🔍 Analyze Image"
    ):

        with st.spinner(
            "Analyzing ultrasound image..."
        ):

            result, mask, measurements = analyze_image(
                image
            )

        if measurements is None:

            st.error(
                "No suitable region was detected."
            )

        else:

            st.success(
                "Automatic analysis completed."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Automatic Segmentation"
                )

                st.image(
                    result,
                    use_container_width=True
                )

            with col2:

                st.subheader(
                    "Geometric Measurements"
                )

                st.metric(
                    "Area",
                    f"{measurements['Area']:.2f} px²"
                )

                st.metric(
                    "Perimeter",
                    f"{measurements['Perimeter']:.2f} px"
                )

                st.metric(
                    "Equivalent Diameter",
                    f"{measurements['Equivalent Diameter']:.2f} px"
                )

                st.metric(
                    "Circularity",
                    f"{measurements['Circularity']:.3f}"
                )

                st.metric(
                    "Bounding Box",
                    f"{measurements['Bounding Box Width']:.0f} × "
                    f"{measurements['Bounding Box Height']:.0f} px"
                )

            st.caption(
                "Measurements are reported in pixels because "
                "no physical image calibration is applied."
            )
'''

with open(
    "app.py",
    "w",
    encoding="utf-8"
) as f:

    f.write(app_code)

print("Corrected app.py created successfully!")
print("Location:", __import__("os").path.abspath("app.py"))

