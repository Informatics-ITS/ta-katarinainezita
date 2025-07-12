import streamlit as st
import numpy as np
import cv2
import albumentations as A
import random
import torch
from ultralytics import YOLO
import os

# Fungsi Preprocessing (Resize)
def preprocess_image(image):
    return cv2.resize(image, (640, 640))

# Fungsi Histogram Equalization
def histogram_equalization(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(image)

# Fungsi Augmentasi Tunggal
def apply_augmentation(image, transform):
    augmented = transform(image=image)
    return augmented['image']

# Fungsi Segmentasi dengan nama kelas
def segment_image(results, class_label, class_name, segmented_path, i, col):
    for result in results:
        if result.masks is not None:
            try:
                boxes = result.boxes.data
                clss = boxes[:, 5]
                indices = torch.where(clss == class_label)
                indices = (indices[0][0].unsqueeze(0),)
                print(f'class: {clss}')
                print(f'indices: {indices}')

                mask_raw = result.masks.cpu().data[indices].numpy().transpose(1, 2, 0)
                mask_3channel = cv2.merge((mask_raw, mask_raw, mask_raw))
                h2, w2, c2 = result.orig_img.shape
                mask = cv2.resize(mask_3channel, (w2, h2))

                hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
                lower_black = np.array([0, 0, 0])
                upper_black = np.array([0, 0, 1])
                mask = cv2.inRange(mask, lower_black, upper_black)
                mask = cv2.bitwise_not(mask)

                masked = cv2.bitwise_and(result.orig_img, result.orig_img, mask=mask)
                resized_mask = cv2.resize(mask, (101, 200))
                file_path = os.path.join(segmented_path, f'{class_name}_segs_{i}.jpg')
                cv2.imwrite(file_path, resized_mask)
                col.image(resized_mask, caption=f"Hasil Segmentasi: {class_name}", width=200)
                print(file_path)
            except Exception as e:
                print('error', e)
                black_image = np.zeros((200, 101), dtype=np.uint8)
                file_path = os.path.join(segmented_path, f'{class_name}_segs_{i}.jpg')
                cv2.imwrite(file_path, black_image)
                col.image(black_image, caption=f"Error in segmentation: {class_name}", width=200)
                print(file_path)
            i += 1
        else:
            black_image = np.zeros((200, 101), dtype=np.uint8)
            file_path = os.path.join(segmented_path, f'{class_name}_segs_{i}.jpg')
            cv2.imwrite(file_path, black_image)
            col.image(black_image, caption=f"Empty: {class_name}", width=200)
            print(file_path)
            i += 1
    return i


# Streamlit App
st.set_page_config(page_title="Image Processing App", layout="wide")

mode = st.radio("Pilih Mode Pemrosesan Citra:", ("Normal", "Histogram Equalization"))
uploaded_file = st.file_uploader("📥 Unggah Citra", type=["png", "jpg", "jpeg"])

# Load YOLO model
model = YOLO('best.pt')  # Ganti path model jika perlu

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state.original_image = image
    st.image(image, caption="🖼️ Citra Asli", width=200)

    # Perform detection
    results = model(image)

    if mode == "Normal":
        st.title("🧪 Normal Mode: Preprocessing Dulu → Augmentasi")
        if st.button("📐 Lakukan Preprocessing (Resize 640x640)"):
            preprocessed = preprocess_image(st.session_state.original_image)
            st.session_state.preprocessed_image = preprocessed
            st.image(preprocessed, caption="📏 Setelah Resize", width=200)

        if "preprocessed_image" in st.session_state:
            st.markdown("### ✨ Hasil Semua Augmentasi")
            augmentation_methods = [
                ("Flip Horizontal", A.HorizontalFlip(p=1.0)),
                ("Rotation", A.Rotate(limit=15, p=1.0, border_mode=0)),
                ("Shear", A.Affine(shear={"x": (-10, 10), "y": (-10, 10)}, p=1.0)),
                ("Hue Shift", A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=0, val_shift_limit=0, p=1.0)),
                ("Saturation", A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=25, val_shift_limit=0, p=1.0)),
                ("Exposure", A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.0, p=1.0)),
                ("Blur", A.GaussianBlur(blur_limit=(1, 3), p=1.0))
            ]
            cols = st.columns(3)
            for idx, (label, transform) in enumerate(augmentation_methods):
                result = apply_augmentation(st.session_state.preprocessed_image, transform)
                with cols[idx % 3]:
                    st.image(result, caption=f"✨ {label}", width=200)

            if "augmented_image" not in st.session_state:
                _, selected_transform = random.choice(augmentation_methods)
                st.session_state.augmented_image = apply_augmentation(
                    st.session_state.preprocessed_image,
                    selected_transform
                )
                st.success("✅ Augmentasi acak otomatis disimpan sebagai `augmented_image`. Siap digunakan untuk segmentasi.")

    elif mode == "Histogram Equalization":
        st.title("📊 Histogram Equalization")
        if st.button("📈 Terapkan Histogram Equalization"):
            equalized_image = histogram_equalization(st.session_state.original_image)
            if len(equalized_image.shape) == 2:
                equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

            equalized_resized = preprocess_image(equalized_image)
            st.session_state.preprocessed_image = equalized_resized
            st.image(equalized_resized, caption="📏 Setelah Resize (Equalized)", width=200)

            st.markdown("### ✨ Hasil Semua Augmentasi")
            augmentation_methods = [
                ("Flip Horizontal", A.HorizontalFlip(p=1.0)),
                ("Rotation", A.Rotate(limit=15, p=1.0, border_mode=0)),
                ("Shear", A.Affine(shear={"x": (-10, 10), "y": (-10, 10)}, p=1.0)),
                ("Hue Shift", A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=0, val_shift_limit=0, p=1.0)),
                ("Saturation", A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=25, val_shift_limit=0, p=1.0)),
                ("Exposure", A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.0, p=1.0)),
                ("Blur", A.GaussianBlur(blur_limit=(1, 3), p=1.0))
            ]
            cols = st.columns(3)
            for idx, (label, transform) in enumerate(augmentation_methods):
                result = apply_augmentation(equalized_resized, transform)
                with cols[idx % 3]:
                    st.image(result, caption=f"✨ {label}", width=200)

            if "augmented_image" not in st.session_state:
                _, selected_transform = random.choice(augmentation_methods)
                st.session_state.augmented_image = apply_augmentation(
                    equalized_resized,
                    selected_transform
                )
                st.success("✅ Augmentasi acak otomatis disimpan sebagai `augmented_image`. Siap digunakan untuk segmentasi.")

    # Tombol segmentasi
    if st.button("🔍 Lakukan Segmentasi"):
        segmented_AB_path = "path_to_AB_segmentation"
        segmented_MC_path = "path_to_MC_segmentation"
        i = 0
        cols = st.columns(2)
        i = segment_image(results, class_label=0, class_name="Tulang Alveolar", segmented_path=segmented_AB_path, i=i, col=cols[0])
        i = segment_image(results, class_label=1, class_name="Kanal Mandibula", segmented_path=segmented_MC_path, i=i, col=cols[1])

else:
    st.info("⬆️ Unggah citra terlebih dahulu.")
