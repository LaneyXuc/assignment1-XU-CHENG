from transformers import pipeline
import streamlit as st
import tempfile

# function part
# img2text
def img2text(image_path):
    caption_generator = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    caption = caption_generator(image_path)[0]["generated_text"]
    return caption

# text2story
def text2story(image_description):
    story_generator = pipeline("text-generation", model="roneneldan/TinyStories-33M")

    prompt = (
        "In a magical little world, "
        + image_description +
        "."
    )

    generated_story = story_generator(
        prompt,
        max_new_tokens=85,
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.25,
        no_repeat_ngram_size=3,
        do_sample=True
    )

    story_text = generated_story[0]["generated_text"]

    return story_text

# text2audio
def text2audio(story_text):
    audio_generator = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_output = audio_generator(story_text)
    return audio_output

# main part
st.set_page_config(page_title="Magic Story App", page_icon="✨")

st.title("Magic Story App")

uploaded_image = st.file_uploader(
    "Upload your picture",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    st.image(
        uploaded_image,
        caption="Selected Picture",
        use_container_width=True
    )

    if st.button("Create Magic Story"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_image.getvalue())
            temp_image_path = temp_file.name

        description = img2text(temp_image_path)

        generated_story = text2story(description)

        st.write(generated_story)

        audio_result = text2audio(generated_story)

        audio_array = audio_result["audio"]

        sample_rate = audio_result["sampling_rate"]

        st.audio(audio_array, sample_rate=sample_rate)
