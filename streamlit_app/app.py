import streamlit as st
from inference import prediction

@st.cache
def convert_csv(file):
    return file.to_csv()

def main():
    st.set_page_config(page_title="KOWOPE-MARK CCDection", layout="wide")
    hide_default_format = """
        <style>
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_default_format, unsafe_allow_html=True)
    st.title("KOWOPE-MART CREDIT CARD CUSTOMER DEFAULT DETECTION")
    
    st.header("Upload File For Inference")
    uploadfile = st.file_uploader("", type="csv", accept_multiple_files=False)
    if uploadfile is not None:
        with st.spinner("prediction is been generated..."):
            predict = prediction(uploadfile)
            st.success("Prediction Generated")
        if st.button("Check Inference"):
            st.dataframe(predict)
        download = convert_csv(predict)
        st.download_button("Download Predictions", download, "prediction.csv", mime="test/csv")
            
if __name__ == "__main__":
    main()