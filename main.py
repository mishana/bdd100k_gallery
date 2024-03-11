from pathlib import Path
from PIL import Image

import streamlit as st
import torchvision.transforms.functional as TF
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split
import torch


data_dir = Path('/Users/misha/data')  # Change this to the directory where you store the data

demo_images_dir = data_dir / 'demo_images'
all_bboxes = demo_images_dir / 'all_bboxes'
suspect_bboxes = demo_images_dir / 'suspect_bboxes'
relabeled_bboxes = demo_images_dir / 'relabeled_bboxes'

all_images_names = [f.name for f in all_bboxes.iterdir()]

@st.cache_data
def get_image_from_folder(folder_path, image_name):
    img = Image.open(folder_path / image_name)
    return img, image_name

# Pagination settings
images_per_page = 4
total_images = len(all_images_names)  # Assuming you want to base pagination on the largest subset
total_pages = (total_images + images_per_page - 1) // images_per_page  # Calculate the total number of pages

st.title('BDD100K Mislabels: Found by Hirundo')

# Initialize or update the current page in the session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0  # Starting from page 0

# Display images for the current page
start_index = st.session_state.current_page * images_per_page
end_index = start_index + images_per_page

col1, col2, col3 = st.columns(3)

with col1:
    st.text('Original: All Bboxes')

with col2:
    st.text('Suspect Bboxes')

with col3:
    st.text('Relabeled Bboxes')

for i in range(start_index, min(end_index, total_images)):
    col1, col2, col3 = st.columns(3)

    img_name = all_images_names[i]
    
    img_all, label_all = get_image_from_folder(all_bboxes, img_name)
    with col1:
        st.image(img_all, caption=f"Label: {label_all}", use_column_width=True)

    img_suspects, label_suspects = get_image_from_folder(suspect_bboxes, img_name)
    with col2:
        st.image(img_suspects, caption=f"Label: {label_suspects}", use_column_width=True)

    img_relabeled, label_relabeled = get_image_from_folder(relabeled_bboxes, img_name)
    with col3:
        st.image(img_relabeled, caption=f"Label: {label_relabeled}", use_column_width=True)

# Centered navigation buttons with conditional disabling
_, col_prev, col_next, _ = st.columns([2,1,1,2])

with col_prev:
    # Disable 'Previous' button if on the first page
    prev_disabled = st.session_state.current_page <= 0
    if st.button('Previous', disabled=prev_disabled, key='prev'):
        st.session_state.current_page -= 1

with col_next:
    # Disable 'Next' button if on the last page
    next_disabled = st.session_state.current_page >= total_pages - 1
    if st.button('Next', disabled=next_disabled, key='next'):
        st.session_state.current_page += 1
