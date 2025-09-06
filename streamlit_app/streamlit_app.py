import streamlit as st
import numpy as np
import matplotlib.image as mpimg  # for image load and processing
import io

from sklearn.cluster import KMeans

# -------------------------------FUNCTIONS-------------------------------------------#
# ------------------01---------------
def preprocess_img(img):

    #--------------------Normalize image -------------------#
    if img.dtype == np.uint8:
        # normalize it bcoz unit8 format is in range [0, 255]
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            # it is rare to have float32 and max val greater than 1, but just in case
            img = img / 255.0

    #-------------------handle alpha image ---------#
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]
        # ... it means keep all preceding dimensions unchanged
        # :3 means on the last axis (i.e for the channels) selects indices (0, 1, 2) (first three channels only)

    # ---------------handling grayscale image ----------#
    elif img.ndim == 2:
        img = np.stack( [img, img, img], axis = -1 )
        # it means make 3 copies of same 2D array and stack them along a new last axis, thus creating (H, W, 3)

    return img

# -----------------02-----------------
def creating_compressed_image(img, k):
    # 01. flatting image for kmeans
    H, W, C = img.shape
    X = img.reshape(-1, 3)

    # 02. fit the k means algo
    kmeans = KMeans(
        n_clusters = k,
        random_state = 11, 
        n_init = 10
    )

    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_

    # compressing the image
    compressed = centroids[labels] # replace each pixel by its centroid color
    compressed = compressed.reshape(H, W, 3)

    return compressed

# ---------------03----------------
def convert_to_image(compressed):
    buffer = io.BytesIO()
    mpimg.imsave(buffer, compressed, format="png")
    buffer.seek(0)  # rewind for reading 
    
    img_file_name = f"compressed_image_{np.random.randint(1, 100)}.png"
    
    return buffer, img_file_name

    
    


# -------------------------------------START ----------------------------------- #
st.title("Image Compression")
st.subheader("(using K Means clustering)")
st.write("")


# jupyter notebook link button


st.link_button(
    # label on the btn
    "**Jupyter Notebook Code**",
    # url
    "https://github.com/shivsharcode/IMAGE-COMPRESSION-mini-project-kmeans-based-/blob/f98f2be0b2748dcf287d06895736e40d6653054a/01.-Developing%20IMage%20Compression%20step-by-step.ipynb", # url
    # help info on hovering
    help="Opens the Jupyter Notebook code on github in a new page",
    # icon
    icon = "🔗" # from google material library
    
)
st.write("")

# K value selection

st.subheader("▶️ Choose K value")
k = st.slider(
    # label
    label = "Lesser the **K**, more the compression",
    min_value= 2,
    max_value = 10, 
    value= 4,
    help = "Lesser the K, more the compression"
)

st.write("")

st.info("**NOTE:**  Lesser the **K** ⬇️, more the **Compression** ⬆️")
st.write("")
st.write("")



# Image uploader
st.subheader("▶️ Upload the Image")
img = st.file_uploader(
    label= "Upload Image",
    type = ["png", "jpg", "jpeg"], 
    # key = "img"
)

if img:
    
    st.image(img, width = 200)
    img = mpimg.imread(img) # converts the img to np array
    img = preprocess_img(img)
    
    # compress button
    if st.button(label = "**COMPRESS IMAGE**", width= "stretch"):
        with st.spinner("Compressing the image .."):

            compressed = creating_compressed_image(img, k)
            compressed = np.clip(compressed, 0, 1)
            
            
            # show images side-by-side using columns
            st.subheader("▶️ Original / Compressed Image")
            st.write("")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img, caption="Original", width=200)
            with col3:
                st.image(np.clip(compressed, 0, 1), caption=f"Compressed with K={k}", width = 200)
                
        # prepare bytes for download -------
        # convert float
        compressed_bytes_io, img_file_name = convert_to_image(compressed)
        
        st.download_button(
            label = "Download image",
            data = compressed_bytes_io,
            file_name= img_file_name,
            mime= "image/png"
        )

        
        
        
        
        
        
                

    




