import streamlit as st
import pandas as pd
import numpy as np

st.title('Hello, Streamlit!')
st.write('This is a simple Streamlit app.')
st.success("Lets get started")

name = st.text_input('Enter your name:')
if st.button("Greet"):
    st.write(f'Hello, {name}!')

df=pd.DataFrame(np.random.randn(10,3),columns=['A','B','C'])
st.line_chart(df)
st.bar_chart(df)

upload_flie=st.file_uploader("Upload a CSV file", type=["csv"])
if upload_flie:
    df=pd.read_csv(upload_file)
    st.dataframe(df)

st.title("Text and Markdown Demo")
st.header("This is a header")
st.subheader("This is a subheader")
st.markdown("*Bold, *Italic, Code, [Link](https://streamlit.io/)")
st.code("for i in range(5): print(i)", language="python")
st.text_input("What's your name?")
st.text_area("Write something...")
st.number_input("Pick a number", min_value=0, max_value=30)
st.slider("Choose a range", 0, 100)
st.selectbox("Select a fruit", ["Apple", "Banana", "Mango"])
st.multiselect("Choose toppings", ["Cheese", "Tomato", "Onion"])
st.radio("Pick one", ["Option A", "Option B"])
st.checkbox("I agree to the terms")

if st.checkbox("Show Details"):
    st.info("Here are more details...")

option = st.radio("Choose view", ["Show Chart", "Show Table"])

if option == "Show Chart":
    st.write("Chart would appear here")
else:
    st.write("Table would appear here")

with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Login")

    if submitted:
        st.success(f"Welcome, {username}!")

        st.image(
    "https://via.placeholder.com/300",
    caption="Sample Image",
    use_container_width=True
)

st.video("https://youtu.be/2F34Iqm-qnU?si=kswwqeu8zH8yy_lO")