import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
from skimage import io
from glob import glob
from tqdm import tqdm

class MaskSegmenter:
    """
    Class for implementing traditional region-based segmentation techniques
    for face mask segmentation.
    """
    
    def __init__(self, data_path, output_path="results/traditional"):
        """
        Initialize the segmenter with dataset paths.
        
        Args:
            data_path: Path to the MFSD dataset
            output_path: Path to save results
        """
        self.data_path = data_path
        self.output_path = output_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "visualizations"), exist_ok=True)
        
        # Paths to images and masks
        self.images_path = os.path.join(data_path, "images")
        self.masks_path = os.path.join(data_path, "masks")
        
        # Get list of image files that have masks
        self.image_files = sorted(glob(os.path.join(self.images_path, "with_mask", "*.jpg")))
        self.mask_files = [os.path.join(self.masks_path, os.path.basename(img)) for img in self.image_files]
        
        print(f"Found {len(self.image_files)} images with masks.")
    
    def preprocess_image(self, image):
        """
        Preprocess the image for segmentation.
        
        Args:
            image: Input image
        
        Returns:
            Preprocessed image
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        return blurred
    
    def threshold_based_segmentation(self, image):
        """
        Segment mask using thresholding in HSV color space.
        
        Args:
            image: Input image
        
        Returns:
            Binary mask of the segmented region
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Define color range for common mask colors (blue, green, white)
        # These ranges can be tuned based on the dataset
        blue_lower = np.array([90, 50, 50])
        blue_upper = np.array([130, 255, 255])
        
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        
        white_lower = np.array([0, 0, 180])
        white_upper = np.array([180, 30, 255])
        
        # Create masks for each color
        blue_mask = cv2.inRange(preprocessed, blue_lower, blue_upper)
        green_mask = cv2.inRange(preprocessed, green_lower, green_upper)
        white_mask = cv2.inRange(preprocessed, white_lower, white_upper)
        
        # Combine masks
        combined_mask = blue_mask | green_mask | white_mask
        
        # Apply morphological operations to improve mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def edge_based_segmentation(self, image):
        """
        Segment mask using edge detection.
        
        Args:
            image: Input image
        
        Returns:
            Binary mask of the segmented region
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny edge detector
        edges = cv2.Canny(blurred, 50, 150)
        
        # Perform dilation to close gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create empty mask
        mask = np.zeros_like(gray)
        
        # Draw filled contours on mask
        # Filter contours by area to remove small noise
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust threshold as needed
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Apply morphological operations to improve mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def hybrid_segmentation(self, image):
        """
        Combine thresholding and edge-based methods for better segmentation.
        
        Args:
            image: Input image
        
        Returns:
            Binary mask of the segmented region
        """
        # Get masks from both methods
        threshold_mask = self.threshold_based_segmentation(image)
        edge_mask = self.edge_based_segmentation(image)
        
        # Combine the masks
        combined_mask = cv2.bitwise_or(threshold_mask, edge_mask)
        
        # Apply morphological operations to improve mask
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find the face region and restrict the mask to it
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_mask = np.zeros_like(gray)
        
        # If face detected, restrict mask to lower part of face
        for (x, y, w, h) in faces:
            # Focus on lower half of face (where mask would be)
            lower_face_y = int(y + 0.4 * h)
            lower_face_h = int(h * 0.6)
            
            face_mask[lower_face_y:y+h, x:x+w] = 255
        
        # If no face detected, use the original mask
        if len(faces) == 0:
            return final_mask
        
        # Combine with face region
        result_mask = cv2.bitwise_and(final_mask, face_mask)
        
        return result_mask
    
    def evaluate_segmentation(self, pred_masks, gt_masks):
        """
        Evaluate segmentation results using various metrics.
        
        Args:
            pred_masks: List of predicted mask images
            gt_masks: List of ground truth mask images
            
        Returns:
            Dictionary of evaluation metrics
        """
        iou_scores = []
        dice_scores = []
        accuracy_scores = []
        
        for pred, gt in zip(pred_masks, gt_masks):
            # Convert to binary masks
            pred_binary = (pred > 0).astype(np.float32).flatten()
            gt_binary = (gt > 0).astype(np.float32).flatten()
            
            # Calculate metrics
            iou = jaccard_score(gt_binary, pred_binary, average='binary', zero_division=1)
            dice = f1_score(gt_binary, pred_binary, average='binary', zero_division=1)
            acc = accuracy_score(gt_binary, pred_binary)
            
            iou_scores.append(iou)
            dice_scores.append(dice)
            accuracy_scores.append(acc)
        
        results = {
            "IoU": np.mean(iou_scores),
            "Dice": np.mean(dice_scores),
            "Accuracy": np.mean(accuracy_scores)
        }
        
        return results
    
    def visualize_results(self, image, gt_mask, pred_mask, idx, save=True):
        """
        Visualize segmentation results.
        
        Args:
            image: Original image
            gt_mask: Ground truth mask
            pred_mask: Predicted mask
            idx: Image index
            save: Whether to save the visualization
        """
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        # Predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_path, "visualizations", f"result_{idx}.png")
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def run_segmentation(self, num_samples=None, visualize=True):
        """
        Run the segmentation pipeline on the dataset.
        
        Args:
            num_samples: Number of samples to process (None for all)
            visualize: Whether to visualize results
            
        Returns:
            Dictionary of evaluation results
        """
        # Use a subset of images if specified
        if num_samples is not None:
            image_files = self.image_files[:num_samples]
            mask_files = self.mask_files[:num_samples]
        else:
            image_files = self.image_files
            mask_files = self.mask_files
        
        # Lists to store results
        pred_masks = []
        gt_masks = []
        
        # Process each image
        for i, (img_path, mask_path) in enumerate(tqdm(zip(image_files, mask_files), total=len(image_files))):
            # Read image and ground truth mask
            image = cv2.imread(img_path)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply segmentation
            pred_mask = self.hybrid_segmentation(image)
            
            # Store results
            pred_masks.append(pred_mask)
            gt_masks.append(gt_mask)
            
            # Visualize results
            if visualize and i % 10 == 0:  # Visualize every 10th image
                self.visualize_results(image, gt_mask, pred_mask, i)
        
        # Evaluate results
        results = self.evaluate_segmentation(pred_masks, gt_masks)
        
        # Print results
        print("\nSegmentation Results:")
        print(f"IoU: {results['IoU']:.4f}")
        print(f"Dice Score: {results['Dice']:.4f}")
        print(f"Accuracy: {results['Accuracy']:.4f}")
        
        # Save results to file
        with open(os.path.join(self.output_path, "results.txt"), "w") as f:
            f.write("Traditional Segmentation Results:\n")
            f.write(f"IoU: {results['IoU']:.4f}\n")
            f.write(f"Dice Score: {results['Dice']:.4f}\n")
            f.write(f"Accuracy: {results['Accuracy']:.4f}\n")
        
        return results


if __name__ == "__main__":
    # Path to the MFSD dataset
    data_path = "MFSD"  # Change this to your dataset path
    
    # Create segmenter
    segmenter = MaskSegmenter(data_path)
    
    # Run segmentation
    results = segmenter.run_segmentation(num_samples=None)  # Set to None to process all images