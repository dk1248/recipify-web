import streamlit as st
import pandas as pd
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize




#Use the full page instead of a narrow central column
st.set_page_config(page_title='Recipify',
                   page_icon = '🍲',
                   layout = 'wide',
                   initial_sidebar_state = 'expanded')

#set background
import base64

@st.cache
def load_image(path):
    with open(path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    return encoded

def image_tag(path):
    encoded = load_image(path)
    tag = f'<img src="data:image/png;base64,{encoded}">'
    return tag

def background_image_style(path):
    encoded = load_image(path)
    style = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    '''
    return style

st.write(background_image_style('back_yellow5.png'), unsafe_allow_html=True)


st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Recipify your picture")
    st.markdown('Upload a picture/screenshot of a cooked meal and get the recipe.')
    uploaded_file = st.file_uploader("Choose a food picture", type = ['png', 'jpg', 'jpeg'])

    recipe_button = st.button("Recipify my picture")


st.sidebar.image('logo6.png', use_column_width=True)


# Space out the maps so the first one is 2x the size of the other three
#col1, col2= st.columns((6, 1))
col1, col2, col3= st.columns((6, 3, 2))

#url = 'http://0.0.0.0:8000'
url = "https://recipify-llestxp3ga-ew.a.run.app"

with col1:
    st.header("Welcome to Recipify")
    st.subheader("See something yummy?")
    st.subheader("🍲 ➟ 📸 ➟ 📜\nTake a picture and find the recipe on the spot!")

    if uploaded_file is not None:
    # display image:
        st.image(uploaded_file, width= 500)

        # To read image file buffer as a PIL Image:
        img = Image.open(uploaded_file)
        img1 = img.convert(mode="RGB")


        # To convert PIL Image to numpy array:
        # Transform image to tensor
        img_tensor = np.array(img1)/255.
        img_tensor = resize(img_tensor, (128, 128))
        img_tensor = tf.convert_to_tensor(img_tensor)

        #img_tensor = tf.image.resize(img_tensor, [128, 128])
        img_tensor = tf.expand_dims(img_tensor, axis = 0)
        #st.write(img_tensor)

        # Check the type of img_tensor:
        # Should output: <class 'numpy.ndarray'> (here tensor!)
        #st.write(type(img_tensor))
        #st.write(img_tensor[0])

        # Check the shape of img_tensor:
        # Should output shape: (height, width, channels)
        #st.write(img_tensor.shape)

        #tranform to tensor
        #bytes_data = uploaded_file.getvalue()
        #img_tensor = tf.io.decode_image(bytes_data, channels=3)
        #img_tensor = img_tensor/255.
        #img_tensor1 = tf.image.resize(img_tensor, [156, 156])
        # Check the type of img_tensor:
        # Should output: <class 'tensorflow.python.framework.ops.EagerTensor'>
        #st.write(type(img_tensor))
        #st.write(img_tensor[0])


    if recipe_button:
    #    entered_items.markdown("**Generate recipe for:** " + items)
        with st.spinner("Generating recipe based on image..."):
            #img = requests.post(url + '/predict')
            img = requests.post(url + '/predict', files = {'img': img_tensor})

            print(img.status_code)
            res = img.json()
            if 'error_message' in res.keys():
                name = res['name']
                recipe_title = f"""
                <span style="color:pink;font-family:sans-serif;font-size:25px;" >{name.capitalize()}</span>
                """
                st.markdown(recipe_title, unsafe_allow_html=True)
                st.error(f"We predict the image you uploaded to be **{res['name']}**. {res['error_message']}")

            else:
                #st.balloons()
                prediction1 = res['prediction1']
                pred_dish = res['pred_dish']
                prediction2 = res['prediction2']
                pred_dish_2 = res['pred_dish_2']
                name = res['name']
                steps = res['steps']
                rating = res['rating']
                ingredients = res['ingredients']
                calories = res['calories']
                total_fat = res['total fat']
                sugar = res['sugar']
                sodium = res['sodium']
                protein = res['protein']
                saturated_fat = res['saturated fat']
                carbohydrates = res['carbohydrates']

                st.warning(f"""
                           With **{round(prediction1*100,2)}%** probability, we predict the image you uploaded to be **{pred_dish}**.
                           With **{round(prediction2*100,2)}%** probability, the uploaded image could also be **{pred_dish_2}**.
                           Maybe, you would like to try out this recipe:
                           """)

                recipe_title = f"""
                <span style="color:pink;font-family:sans-serif;font-size:25px;" >{name.capitalize()}</span>
                """
                st.markdown(recipe_title, unsafe_allow_html=True)
                step = steps.split(',')
                sp = ''

                n = 3
                chunks = [step[i:i+n] for i in range(0, len(step), n)]
                for x in chunks:
                   # x = x.split()
                    s = ''.join(x)
                    sp += "1. " + s + "\n"
                st.markdown(sp)

                nutrition_title = '<p style="font-family:sans-serif; color:Pink; font-size: 25px;">Nutritional Information</p>'
                st.markdown(nutrition_title, unsafe_allow_html=True)
                y = [float(total_fat), float(saturated_fat), float(protein), float(carbohydrates), float(sugar), float(sodium)]
                x = ['total fat in g', 'saturated fat in g', 'protein in g', 'carbohydrates in g', 'sugar in g', 'sodium in mg']
                sns.set(rc={'axes.facecolor': (0,0,0,0), 'figure.facecolor':(0,0,0,0)})
                #sns.set_style("darkgrid")
                fig, ax = plt.subplots() #solved by add this line
                ax = sns.barplot(x=x, y= y, color="sandybrown") #.set(title='Nutritional Values') palette = 'rocket'
                ax.grid(axis = 'y', color='#5C5C5E', linewidth=0.5)
                plt.xticks(rotation=45)
                st.pyplot(fig)


                with col2:
                    st.markdown('#')
                    st.markdown('#')
                    st.markdown('#')
                    st.markdown('#')
                    st.markdown('#')
                    st.markdown('#')
                    st.markdown('#')
                    rating_title = '<p style="font-family:sans-serif; color:Pink; font-size: 25px;">Rating</p>'
                    st.markdown(rating_title, unsafe_allow_html=True)
                    if rating >0 and rating <2:
                        st.markdown("### ★☆☆☆☆")
                    if rating >=2 and rating <3:
                        st.markdown("### ★★☆☆☆")
                    if rating >=3 and rating <4:
                        st.markdown("### ★★★☆☆")
                    if rating >=4 and rating <5:
                        st.markdown("### ★★★★☆")
                    if rating >=5:
                        st.markdown("#### ★★★★★")
                    #st.markdown(rating)


                    ingredients_title = '<p style="font-family:sans-serif; color:Pink; font-size: 25px;">Ingredients</p>'
                    st.markdown(ingredients_title, unsafe_allow_html=True)
                    lst = ingredients.split(',')
                    s = ''
                    for i in lst:
                        s += "- " + i + "\n"
                    st.markdown(s)

                    calories_title = '<p style="font-family:sans-serif; color:Pink; font-size: 25px;">Calories</p>'
                    st.markdown(calories_title, unsafe_allow_html=True)
                    st.markdown(f'**{calories}**')


                    #nutri =  st.checkbox('Show nutritional information')
                    #if nutri:
                    #    st.write(sugar)
