
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="SonoVision",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Model
# -----------------------------

class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )

        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# -----------------------------
# Load model
# -----------------------------

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


model = load_model()


# -----------------------------
# Functions
# -----------------------------

def analyze_image(image):

    original = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    resized = cv2.resize(
        gray,
        (128, 128)
    )

    input_image = resized.astype(np.float32) / 255.0

    tensor = torch.tensor(
        input_image,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        prediction = torch.sigmoid(
            model(tensor)
        )

    mask = (
        prediction.squeeze().numpy() > 0.5
    ).astype(np.uint8)

    mask = cv2.resize(
        mask,
        (original.shape[1], original.shape[0]),
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

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter > 0:

        circularity = (
            4 * np.pi * area
            / (perimeter ** 2)
        )

    else:
        circularity = 0

    equivalent_diameter = (
        np.sqrt(4 * area / np.pi)
        if area > 0 else 0
    )

    x, y, w, h = cv2.boundingRect(contour)

    # Overlay
    overlay = original.copy()

    overlay[mask > 0] = (
        255,
        0,
        0
    )

    result = cv2.addWeighted(
        original,
        0.7,
        overlay,
        0.3,
        0
    )

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
        "Equivalent Diameter": equivalent_diameter,
        "Circularity": circularity,
        "Bounding Box Width": w,
        "Bounding Box Height": h
    }

    return result, mask, measurements


# -----------------------------
# Interface
# -----------------------------

st.title("🩺 SonoVision")

st.subheader(
    "Ultrasound Image Analysis Using Computer Vision"
)

st.write(
    "SonoVision automatically segments the lumen "
    "region in carotid ultrasound images and provides "
    "basic geometric measurements."
)

st.info(
    "Research and visualization project only. "
    "This system is not intended for medical diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload an ultrasound image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Ultrasound Image")

    st.image(
        image,
        use_container_width=True
    )

    if st.button("Analyze Image"):

        with st.spinner("Analyzing ultrasound image..."):

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
                    "Measurements"
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

            st.caption(
                "Measurements are reported in pixels because "
                "no physical image calibration is applied."
            )
