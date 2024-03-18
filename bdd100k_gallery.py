
import pandas as pd
import streamlit as st
from streamlit_image_comparison import image_comparison

st.set_page_config(layout="wide")

# st.markdown(
#     """
#     <style>
#     .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
#     .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
#     .viewerBadge_text__1JaDK {
#         display: none;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

images_ext = 'webp'

@st.cache_data
def get_all_filtered_images_names():
    images_metadata = pd.read_csv('images_metadata.csv')
    all_images_names = [f'{f_idx}.{images_ext}' for i, f_idx in enumerate(images_metadata['frame_idx']) if not images_metadata.iloc[i]['is_filter']]
    return all_images_names

@st.cache_data
def get_images_names_to_sorted_idx():
    images_metadata = pd.read_csv('images_metadata.csv')
    all_images_names = [f'{f_idx}.{images_ext}' for i, f_idx in enumerate(images_metadata['frame_idx'])]
    images_names_to_sorted_idx = {img_name: i for i, img_name in enumerate(sorted(all_images_names))}
    return images_names_to_sorted_idx

@st.cache_data
def get_dataset():
    from datasets import load_dataset

    dataset = load_dataset("odnurih/bdd100k-mislabels", split="train")
    return dataset

@st.cache_data
def get_image_from_folder(data_subset, image_name):
    images_names_to_sorted_idx = get_images_names_to_sorted_idx()
    sorted_idx = images_names_to_sorted_idx[image_name]

    dataset = get_dataset()
    label_names = dataset.features['label'].names
    label_idx = label_names.index(data_subset)
    dataset_len = len(dataset)

    in_dataset_idx = label_idx * dataset_len // len(label_names) + sorted_idx

    img = get_dataset()[in_dataset_idx]['image']
    image_name = image_name.split('.')[0]
    return img, image_name

# Pagination settings
images_per_page = 2
total_images = len(get_all_filtered_images_names())  # Assuming you want to base pagination on the largest subset
total_pages = (total_images + images_per_page - 1) // images_per_page  # Calculate the total number of pages

_, col_center, _ = st.columns([1, 2, 1])
with col_center:
    st.title('BDD100K Mislabels: Found by Hirundo')

# Initialize or update the current page in the session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0  # Starting from page 0

def enter_page_num():
    st.session_state.current_page = st.session_state.page_num - 1

def increment_page():
    st.session_state.current_page += 1

def decrement_page():
    st.session_state.current_page -= 1

def prev_disabled():
    return st.session_state.current_page <= 0

def next_disabled():
    return st.session_state.current_page >= total_pages - 1

# Display images for the current page
start_index = st.session_state.current_page * images_per_page
end_index = start_index + images_per_page

col1, col2, col3 = st.columns(3, gap='large')

with col1:
    st.text('Original: All Bboxes')

with col2:
    st.text('Suspect Bboxes')

with col3:
    st.text('Automatically Relabeled Bboxes')

for i in range(start_index, min(end_index, total_images)):
    # col1, col2, col3 = st.columns(3, gap='large')

    img_name = get_all_filtered_images_names()[i]
    
    img_all, label_all = get_image_from_folder('all_bboxes', img_name)
    with col1:
        st.image(img_all, caption=f"Index: {label_all}", use_column_width=True)
        # st.image(img_all, use_column_width=True, caption="")

    img_suspects, label_suspects = get_image_from_folder('suspect_bboxes', img_name)
    with col2:
        st.image(img_suspects, caption=f"Index: {label_suspects}", use_column_width=True)
        # st.image(img_suspects, use_column_width=True)

    img_relabeled, label_relabeled = get_image_from_folder('relabeled_bboxes', img_name)
    with col3:
        st.image(img_relabeled, caption=f"Index: {label_relabeled}", use_column_width=True)
        # st.image(img_relabeled, use_column_width=True, caption=" ")

# Centered navigation buttons with conditional disabling
_, col_page_num, _ = st.columns(3, gap='large')

# with col_prev:
#     st.button('Previous', disabled=prev_disabled(), key='prev', on_click=decrement_page)

# with col_next:
#     st.button('Next', disabled=next_disabled(), key='next', on_click=increment_page)

# with col_prev:
#     image_comparison(
#         img_suspects,
#         img_relabeled,
#         label1="Suspect Bboxes",
#         label2="Automatically Relabeled Bboxes",
#     )

with col_page_num:
    num = st.number_input(label="Enter page num:", on_change=enter_page_num, value=1, step=1, label_visibility='visible', min_value=1, max_value=total_pages, key='page_num')

