import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Function to perform image manipulation
def perform_image_manipulation(image, manipulation_type, params):
    if image is None:
        st.error("Error: Image not loaded. Please upload a valid image.")
        return None

    if manipulation_type == "Original":
        return image
    elif manipulation_type == "Resize":
        new_width, new_height = params['new_width'], params['new_height']
        resized_img = cv2.resize(image, (new_width, new_height))
        return resized_img
    elif manipulation_type == "Blur":
        kernel_size = params['kernel_size']
        blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blur
    elif manipulation_type == "Canny Edge Detection":
        min_threshold, max_threshold = params['min_threshold'], params['max_threshold']
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, min_threshold, max_threshold)
        return canny
    elif manipulation_type == "Dilate Edges":
        kernel_size = params['kernel_size']
        iterations = params['iterations']
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(image, kernel, iterations=iterations)
        return dilated
    elif manipulation_type == "Crop":
        x1, y1, x2, y2 = params['x1'], params['y1'], params['x2'], params['y2']
        cropped = image[y1:y2, x1:x2]
        return cropped
    elif manipulation_type == "Translate":
        x_translation, y_translation = params['x_translation'], params['y_translation']
        M = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
        translated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return translated
    elif manipulation_type == "Rotate":
        angle, scale = params['angle'], params['scale']
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return rotated
    elif manipulation_type == "Flip Horizontal":
        flipped_horizontal = cv2.flip(image, 1)
        return flipped_horizontal
    elif manipulation_type == "Flip Vertical":
        flipped_vertical = cv2.flip(image, 0)
        return flipped_vertical
    elif manipulation_type == "Flip Both":
        flipped_both = cv2.flip(image, -1)
        return flipped_both
    elif manipulation_type == "Histogram":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.title('Grayscale Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.xlim([0, 256])
        st.pyplot()
        return image
    elif manipulation_type == "Convolution":
        conv_matrix = params['conv_matrix']
        conv_result = convolve2d(image[:, :, 0], conv_matrix, mode='same', boundary='wrap')
        plt.subplot(1, 2, 1)
        plt.imshow(image)  # for colored
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(conv_result, cmap='gray')
        plt.title('Convolution Result')
        plt.tight_layout()
        st.pyplot()
        return image
    elif manipulation_type == "Face Detection":
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        # Ensure the cascade classifier loaded successfully
        if face_cascade.empty():
            st.error("Error: Unable to load the face cascade classifier.")
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        # Create a copy of the original image for drawing rectangles
        image_with_faces = image.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the image with faces
        st.image(image_with_faces, caption="Image with Face Detection", use_column_width=True, channels="BGR")
        return image
    else:
        return image

# Streamlit app for image manipulation
def perform_image_manipulation_app():
    st.title("Image Manipulation App")

    # Sidebar for user input
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Convert image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display original image
        st.image(image_rgb, caption="Original Image", use_column_width=True, width=150)

        # Sidebar options
        manipulation_type = st.sidebar.selectbox("Choose manipulation type", ["Original", "Resize", "Blur", "Canny Edge Detection", "Dilate Edges", "Crop", "Translate", "Rotate", "Flip Horizontal", "Flip Vertical", "Flip Both", "Histogram", "Convolution", "Face Detection"])

        # Parameters for selected manipulation
        params = {
            'new_width': 0,
            'new_height': 0,
            'kernel_size': 0,
            'min_threshold': 0,
            'max_threshold': 0,
            'iterations': 0,
            'x1': 0,
            'y1': 0,
            'x2': 0,
            'y2': 0,
            'x_translation': 0,
            'y_translation': 0,
            'angle': 0,
            'scale': 0,
            'conv_matrix': None,
        }

        # Update params based on the selected manipulation type
        if manipulation_type == "Resize":
            params['new_width'] = st.sidebar.slider("New Width", 1, 2000, 800)
            params['new_height'] = st.sidebar.slider("New Height", 1, 2000, 600)
        elif manipulation_type == "Blur":
            params['kernel_size'] = st.sidebar.slider("Kernel Size", 1, 31, 7, step=2)
        elif manipulation_type == "Canny Edge Detection":
            params['min_threshold'] = st.sidebar.slider("Min Threshold", 0, 255, 125)
            params['max_threshold'] = st.sidebar.slider("Max Threshold", 0, 255, 175)
        elif manipulation_type == "Dilate Edges":
            params['kernel_size'] = st.sidebar.slider("Kernel Size", 1, 31, 3, step=2)
            params['iterations'] = st.sidebar.slider("Iterations", 1, 10, 3)
        elif manipulation_type == "Crop":
            params['x1'] = st.sidebar.slider("X1", 0, image.shape[1], 0)
            params['y1'] = st.sidebar.slider("Y1", 0, image.shape[0], 0)
            params['x2'] = st.sidebar.slider("X2", 0, image.shape[1], image.shape[1])
            params['y2'] = st.sidebar.slider("Y2", 0, image.shape[0], image.shape[0])
        elif manipulation_type == "Translate":
            params['x_translation'] = st.sidebar.slider("X Translation", -200, 200, 0)
            params['y_translation'] = st.sidebar.slider("Y Translation", -200, 200, 0)
        elif manipulation_type == "Rotate":
            params['angle'] = st.sidebar.slider("Angle", -180, 180, 45)
            params['scale'] = st.sidebar.slider("Scale", 0.1, 2.0, 1.0)
        elif manipulation_type == "Convolution":
            params['kernel_size'] = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
            params['conv_matrix'] = np.ones((params['kernel_size'], params['kernel_size']), np.float32) / (params['kernel_size'] * params['kernel_size'])
        elif manipulation_type == "Face Detection":
            # No additional parameters needed for face detection
            pass

        # Perform image manipulation
        manipulated_image = perform_image_manipulation(image, manipulation_type, params)

        # Convert manipulated image from BGR to RGB
        manipulated_image_rgb = cv2.cvtColor(manipulated_image, cv2.COLOR_BGR2RGB)

        # Use columns to display original and manipulated images side by side
        col1, col2 = st.columns(2)
        col1.image(image_rgb, caption="Original Image", use_column_width=True, width=150)
        col2.image(manipulated_image_rgb, caption=f"{manipulation_type} Image", use_column_width=True, width=150)


if __name__ == "__main__":
    perform_image_manipulation_app()
