import torch
from PIL import Image
import streamlit as st
from ex import get_encoder, Decoder, stylize_adain
# --- Streamlit UI ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_file = st.file_uploader("Загрузите контент-изображение", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Загрузите стиль-изображение", type=["jpg", "jpeg", "png"])
alpha = st.slider("Интенсивность стиля (alpha)", 0.0, 1.0, 1.0, step=0.05)

if content_file and style_file:
    content_image = Image.open(content_file).convert("RGB")
    style_image = Image.open(style_file).convert("RGB")
    st.image(content_image, caption="Контент", width=455)
    st.image(style_image, caption="Стиль", width=455)

    if st.button("Перенести стиль AdaIN"):
        with st.spinner("Переносим стиль..."):
            encoder = get_encoder().to(device)
            decoder = Decoder().to(device)
            decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
            decoder.eval()
            output_image = stylize_adain(content_image, style_image, decoder, encoder, alpha)
            st.image(output_image, caption="Результат", width=455)
            st.success("Готово!")
