import streamlit as st
from image_manipulation import perform_image_manipulation_app
from data_exploration import perform_data_exploration
from regex_operations import perform_regex_operations
from audio_manipulation import perform_audio_manipulation

def main():
    st.title("Streamlit Image Manipulation App")

    # Navigation bar
    menu = ["Home","Data Exploration", "Image Manipulation", "Regex Operations","Audio Manipulation"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # Page content based on the selected choice
    if choice == "Home":
        st.write("Welcome to Home App!")

    elif choice == "Data Exploration":
        perform_data_exploration()

    elif choice == "Image Manipulation":
        perform_image_manipulation_app()

    elif choice == "Regex Operations":
        perform_regex_operations()

    elif choice == "Audio Manipulation":  # Add a new option for audio manipulation
        perform_audio_manipulation()
    

# Run the Streamlit app
if __name__ == "__main__":
    main()
