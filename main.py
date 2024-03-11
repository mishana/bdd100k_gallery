import streamlit as st
import torchvision.transforms.functional as TF
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split
import torch

# Assuming CIFAR100 is already downloaded and available
cifar100 = CIFAR100(root='~/data', train=True, download=True)

# Splitting CIFAR100 into three subsets
lengths = [int(len(cifar100) * 0.3), int(len(cifar100) * 0.3), len(cifar100) - 2 * int(len(cifar100) * 0.3)]
a, b, c = random_split(cifar100, lengths)

# Function to convert a PyTorch tensor to a PIL image
def get_image_from_dataset(dataset, idx):
    img, label = dataset[idx]
    if isinstance(img, torch.Tensor):
        img = TF.to_pil_image(img)
    return img, label

# Pagination settings
images_per_page = 4
total_images = max(len(a), len(b), len(c))  # Assuming you want to base pagination on the largest subset
total_pages = (total_images + images_per_page - 1) // images_per_page  # Calculate the total number of pages

# Initialize or update the current page in the session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0  # Starting from page 0

# Display images for the current page
start_index = st.session_state.current_page * images_per_page
end_index = start_index + images_per_page

for i in range(start_index, min(end_index, total_images)):
    col1, col2, col3 = st.columns(3)
    
    if i < len(a):
        img_a, label_a = get_image_from_dataset(a, i)
        with col1:
            st.image(img_a, caption=f"Label: {a.dataset.classes[label_a]}", use_column_width=True)
    
    if i < len(b):
        img_b, label_b = get_image_from_dataset(b, i)
        with col2:
            st.image(img_b, caption=f"Label: {b.dataset.classes[label_b]}", use_column_width=True)
    
    if i < len(c):
        img_c, label_c = get_image_from_dataset(c, i)
        with col3:
            st.image(img_c, caption=f"Label: {c.dataset.classes[label_c]}", use_column_width=True)

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
