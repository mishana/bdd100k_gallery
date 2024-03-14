from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Image
import os

def load_dataset_from_directory(root_directory):
    """Load images, labels, and filenames from subdirectories."""
    # Define the features of your dataset
    features = Features({
        'image': Image(),
        'label': ClassLabel(names=['all_bboxes', 'relabeled_bboxes', 'suspect_bboxes']),
        'filename': Value('string'),
    })
    
    # Initialize lists to store your data
    images, labels, filenames = [], [], []
    
    # Loop through each subdirectory in the root directory
    for label in ['all_bboxes', 'relabeled_bboxes', 'suspect_bboxes']:
        directory_path = os.path.join(root_directory, label)
        for filename in sorted(os.listdir(directory_path)):
            if filename.endswith(".webp"):  # Adjust file types as needed
                filepath = os.path.join(directory_path, filename)
                
                # Add data to lists
                images.append(filepath)
                labels.append(label)
                filenames.append(filename)
    
    # Create a Dataset from the lists of data
    dataset = Dataset.from_dict({
        'image': images,
        'label': labels,
        'filename': filenames,
    }, features=features)
    
    return dataset

# Assuming 'root_directory' is the path to your folder 'a'
root_directory = './'
dataset = load_dataset_from_directory(root_directory)

# You can now access your dataset with images, labels, and filenames
print(dataset)
