import requests
import json
import gradio as gr

# Build some prerequisite objects to use later
url = "https://api.aimlapi.com/generate"
apiKey = "3a008840130544c4997080760a658cb4"
headers = {
    "Authorization": f"Bearer {apiKey}",
    "Content-Type": "application/json",
}

prompt = "Create a relaxing ambient music track"

def main(pr):
    payload = {
        "prompt": pr,
        "make_instrumental": True,
        "wait_audio": True
    }

    print(f"Attempting to generate song with prompt: {pr}")
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
    for chunk in audio_response.iter_content(chunk_size=256):
        if chunk:
            yield chunk


if __name__ == "__main__":
    app = gr.Interface(fn=main, inputs=gr.Textbox(label='Prompt'), outputs=gr.Audio(streaming=True))
    app.launch()
