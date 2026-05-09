import matplotlib.pyplot as plt
import cv2
from config import EDA_OUTPUT_DIR

class Visualizer:
    """Utilities for visualizing data and results."""

    def __init__(self):
        pass

    def plot_class_distribution(self, df, filename: str = 'class_distribution.png'):
        """Plot the distribution of classes (labels)."""
        if 'label' not in df.columns:
            raise ValueError("DataFrame must have a 'label' column.")
        
        plt.figure(figsize=(10, 6))
        df['label'].value_counts().plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Species')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        output_path = EDA_OUTPUT_DIR / filename
        plt.savefig(output_path)
        plt.close()
        print(f"Class distribution plot saved to {output_path}")

    def display_sample_images(self, df, num_samples: int = 5, filename: str = 'sample_images.png'):
        """Display a grid of sample images from each class."""
        labels = df['label'].unique()
        fig, axes = plt.subplots(len(labels), num_samples, figsize=(num_samples * 2, len(labels) * 2))
        
        for i, label in enumerate(labels):
            class_df = df[df['label'] == label].head(num_samples)
            for j, (_, row) in enumerate(class_df.iterrows()):
                if j >= num_samples:
                    break
                img = cv2.imread(row['file_path'])
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(label)
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        output_path = EDA_OUTPUT_DIR / filename
        plt.savefig(output_path)
        plt.close()
        print(f"Sample images saved to {output_path}")