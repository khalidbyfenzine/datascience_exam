# regex_operations.py
import streamlit as st
import re

def perform_regex_operations():
    st.header("Regex Operations")

    # File upload section
    text_file = st.file_uploader("Upload Text File", type=["txt"])
    
    # Input text area
    st.write("### or")
    input_text = st.text_area("Type or Paste Text")

    if text_file is not None:
        text_content = text_file.read().decode("utf-8")
    else:
        text_content = input_text

    if text_content:
        st.write("### Text Content:")
        st.write(text_content)

        # Sidebar controls
        st.sidebar.subheader("Regex Options")

        # Extract email addresses using regex
        if st.sidebar.checkbox("Extract Email Addresses"):
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_content)
            st.write("### Extracted Email Addresses:")
            st.write(emails)

        # Find simple string matches
        if st.sidebar.checkbox("Find Simple String Matches"):
            search_string = st.sidebar.text_input("Enter string to search:")

            # Option to display all words containing the specified character
            if search_string.isalpha() and st.sidebar.checkbox("Show All Words with Character"):
                words = re.findall(r'\b\w*' + re.escape(search_string) + r'\w*\b', text_content, flags=re.IGNORECASE)
                st.subheader("All Words with Character:")
                st.write(words)
            else:
                matches = re.finditer(search_string, text_content, flags=re.IGNORECASE)
                st.write(f"### Matches for '{search_string}':")
                st.write([match.group() for match in matches])

        # Display length of text
        if st.sidebar.checkbox("Show Length of Text"):
            st.write(f"### Length of Text:")
            st.write(len(text_content))

if __name__ == "__main__":
    perform_regex_operations()
