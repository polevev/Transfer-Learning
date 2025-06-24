import streamlit as st
import torch
from PIL import Image
from prjct import image_loader, run_style_transfer, tensor_to_pil

st.title("Neural Style Transfer")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
content_file = st.file_uploader("Загрузите контент-изображение", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Загрузите стиль-изображение", type=["jpg", "jpeg", "png"])
num_steps = st.slider("Количество итераций", 1, 500, 10)

if content_file and style_file:
    content_image = Image.open(content_file).convert("RGB")
    style_image = Image.open(style_file).convert("RGB")
    st.image(content_image, caption="Контент", width=1076)
    st.image(style_image, caption="Стиль", width=1076)
    if st.button("Перенести стиль"):
        with st.spinner("Переносим стиль..."):
            content_tensor = image_loader(content_image)
            style_tensor = image_loader(style_image)
            output_tensor = run_style_transfer(content_tensor, style_tensor, num_steps=num_steps)
            output_image = tensor_to_pil(output_tensor)
            st.image(output_image, caption="Результат", width=768)
            st.success("Готово!")