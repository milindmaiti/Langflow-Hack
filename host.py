import requests
import base64
from elevenlabs.client import ElevenLabs
import gradio as gr
import requests
import json
import base64
from PIL import Image
import io

import os
from dotenv import load_dotenv

import re
load_dotenv()

elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
effects_path = "effects.mp3"
song_path = "audio_file.mp3"
# Assuming you have an image object called 'img'


# If the image is base64 encoded in the text file


system_prompt = "Based on the scenery in this image, generate a description of the sounds that you think could be heard if you were in the image. For example, if there is a river or stream in the background, please provide a detailed description of the sound of water flowing down. The description of the sounds in the image should make the user feel like they are a part of the image, and reliving the moment. If there is a user prompt, ensure to include those instructions as well. Make sure to limit your response to 20 words."

def generate_sound_effect(text: str, output_path: str):

    result = elevenlabs.text_to_sound_effects.convert(
        text=text,
        duration_seconds=10,  
        prompt_influence=0.3,  
    )

    for chunk in result:
        yield chunk

def generate_and_download_music(prompt):
    url = "https://api.aimlapi.com/generate"
    headers = {
        "Authorization": f"Bearer {os.getenv('AIML_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "make_instrumental": True,
        "wait_audio": True
    }

    print(f"Attempting to generate song with prompt: {prompt}")
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response_json = response.json()
    print(response_json)
    song_id1 = response_json[0]['id']
    url1 = response_json[0]['audio_url']

    song_id2 = response_json[1]['id']
    url2 = response_json[1]['audio_url']

    # Print the important data from the first response
    print(f"ID (1): {song_id1}")
    print(f"Streaming URL - Song 1: {url1}")

    print(f"ID (2): {song_id2}")
    print(f"Streaming URL - Song 2: {url2}")

    # Stream the audio content
    audio_response = requests.get(url2, stream=True)

    for chunk in audio_response.iter_content(chunk_size=4096):
        if chunk:
            yield chunk

def api_call(img, txt):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    url = "https://api.imgbb.com/1/upload"
    params = {
        'expiration': '600',
        'key': f'{os.getenv("IMGUR_KEY")}'
    }
    files = {
        "image": ("image.jpg", img_byte_arr, "image/jpeg")
    }
    

    img_upload_response = requests.post(url, params=params, files=files)
    img_upload_response = img_upload_response.json()
    image_path = img_upload_response['data']['url']

    url = "https://api.aimlapi.com/chat/completions"
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {os.getenv('AIML_KEY')}"

    }
    payload = {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": f"{system_prompt} : User prompt: {txt}"},
            {"type": "image_url", "image_url": {"url": f"{image_path}"}}
          ]
        }
      ],
      "max_tokens": 300
    }

    response = requests.post(url, headers=headers, json=payload)

    response_json = response.json()

    content = response_json['choices'][0]['message']['content']
    return content 

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            user_img = gr.Image(type="pil")
            with gr.Row():
                btn_left = gr.Button(value="<", size='sm')
                btn_right = gr.Button(value=">", size='sm')

            user_txt = gr.Textbox(label="Enter Text")
            submit_btn = gr.Button(value="Submit")
        with gr.Column(scale=1):
            txt2 = gr.Textbox(label = "API Response")
            ad1 = gr.Audio(streaming=True)
            ad2 = gr.Audio(streaming=True)

    submit_btn.click(
            fn=api_call,
            inputs=[user_img, user_txt],
            outputs=txt2
    ).then(
        fn=generate_sound_effect,
        inputs=txt2,
        outputs=ad2
    ).then(
        fn=generate_and_download_music,
        inputs=txt2,
        outputs=ad1
    )

# demo = gr.Interface(
    # fn=api_call,
    # inputs=[gr.Image(type="pil"), gr.Textbox(label="Enter Text")],
    # outputs=[gr.Textbox(label = "API Response"), gr.Audio(type="filepath"), gr.Audio(type="filepath")]
# )

demo.launch()
