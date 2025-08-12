import gradio as gr
import sounddevice as sd
import numpy as np
import whisper
import threading
import queue
from datetime import datetime
from docx import Document
import os

# ===== Load Whisper model =====
print("Loading Whisper model...")
model = whisper.load_model("small")
print("Model loaded.\n")

audio_queue = queue.Queue()
transcription = []
stop_event = threading.Event()
stream = None
live_callback = None  # for UI updates


# ===== Audio Callback =====
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy().flatten().astype(np.float32))


# ===== Transcriber Thread =====
def transcriber():
    buffer = np.zeros((0,), dtype=np.float32)
    while not stop_event.is_set() or not audio_queue.empty():
        try:
            audio_chunk = audio_queue.get(timeout=0.1)
            buffer = np.concatenate((buffer, audio_chunk))

            if len(buffer) >= 16000 * 3:  # process every ~3 seconds
                segment = buffer.copy()
                buffer = np.zeros((0,), dtype=np.float32)
                result = model.transcribe(segment, fp16=False, language="en")
                text = result.get("text", "").strip()
                if text:
                    transcription.append(text)
                    if live_callback:
                        live_callback(" ".join(transcription))
        except queue.Empty:
            continue


# ===== Recording Controls =====
def start_recording():
    global stream, transcription, stop_event
    transcription = []
    stop_event.clear()
    stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=audio_callback)
    stream.start()
    threading.Thread(target=transcriber, daemon=True).start()
    return "🎙️ Listening..."


def stop_recording():
    global stream, stop_event
    stop_event.set()
    if stream:
        stream.stop()
        stream.close()
    if transcription:
        doc = Document()
        doc.add_paragraph(" ".join(transcription))
        filename = f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        doc.save(filename)
        return " ".join(transcription), filename
    else:
        return "", None


# ===== UI =====
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")) as demo:
    with gr.Row():
        gr.Markdown(
            "<h1 style='text-align:center; color:#4B0082;'>🎤 Real-time Whisper Speech-to-Text</h1>"
            "<p style='text-align:center; color:gray;'>Powered by OpenAI Whisper & Python</p>"
        )

    with gr.Row():
        live_text = gr.Textbox(label="📜 Live Transcription", lines=12, placeholder="Your speech will appear here...",
                               show_label=True)

    with gr.Row():
        start_btn = gr.Button("🚀 Start Recording", variant="primary", size="lg")
        stop_btn = gr.Button("🛑 Stop & Save", variant="stop", size="lg")

    with gr.Row():
        file_output = gr.File(label="📂 Download Transcription DOCX", file_types=[".docx"])


    def set_live_callback(text):
        live_text.value = text


    live_callback = set_live_callback

    start_btn.click(start_recording, outputs=live_text)
    stop_btn.click(stop_recording, outputs=[live_text, file_output])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)