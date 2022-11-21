from PIL import Image
import imagehash
import streamlit as st

st.title("Image Hash Comparison")

image_file1 = st.file_uploader("Image 1", type=["png","jpg","jpeg"], key="img1")
if image_file1 is not None:
    st.image(image_file1)

image_file2 = st.file_uploader("Image 2", type=["png","jpg","jpeg"], key="img2")
if image_file2 is not None:
    st.image(image_file2)

if image_file1 is not None and image_file2 is not None:

    Hash1 = imagehash.dhash(Image.open(image_file1.name))
    st.write('Picture 1: ' + str(Hash1))

    # Create the Hash Object of the second image
    Hash2 = imagehash.dhash(Image.open(image_file2.name))
    st.write('Picture 2: ' + str(Hash2))

    st.write('distance: '+ str(Hash1 - Hash2))
    # Compare hashes to determine whether the pictures are the same or not
    if(abs(Hash1 - Hash2) < 25):
        st.write("The pictures are perceptually the same !")
    else:
        st.write("The pictures are different")