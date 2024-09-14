import pyaudio
import google.generativeai as genai
import os
from google.cloud import speech
from google.cloud import texttospeech
import tempfile
import pygame
import random

# Set environment variable for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(), 'keyTTS.json')

# Configure API key for Google Generative AI
genai.configure(api_key='AIzaSyDp_HkDlkVOpwhENLvxqD8jBmOBkNKHnQo')

# Set up Google Cloud Speech clients
client = speech.SpeechClient.from_service_account_file('key.json')

# Audio recording parameters
RATE = 44100
CHUNK = int(RATE / 10)  # 100ms

def generate_audio_chunks(stream):
    """Generator that yields audio chunks from the microphone."""
    while True:
        data = stream.read(CHUNK)
        if not data:
            break
        yield speech.StreamingRecognizeRequest(audio_content=data)

def get_gemini_response(text):
    """Send text to Gemini AI and get the response using the SDK."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(text)
    cleaned_response = clean_up_text(response.text)
    return cleaned_response

def clean_up_text(text):
    """Clean up the response text to make it sound more natural."""
    text = text.replace("I am a large language model", "")
    text = text.replace("As an AI", "")
    text = text.replace("My programming", "")
    text = text.replace("You know", "")
    text = text.replace("Right", "")
    text = text.replace("*", "")
    text = text.replace("", "")
    text = text.replace("!!", "!")
    return text.strip()

def synthesize_and_play_audio(text):
    """Synthesize text to speech and play the audio."""
    # Initialize the Text-to-Speech client
    tts_client = texttospeech.TextToSpeechClient()

    # Set the text input
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Select the voice
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name='en-US-Wavenet-F'
    )

    # Configure the audio output
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,  # Use LINEAR16 for WAV
        speaking_rate=0.9,  # Slightly slower speech
        pitch=0.0
    )

    # Perform the text-to-speech request
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # Use a temporary file for the audio output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
        output_file = temp_wav_file.name
        with open(output_file, "wb") as out:
            out.write(response.audio_content)

    print(f"Audio content written to file '{output_file}'")

    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()

    # Wait until the audio finishes playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def listen_print_loop(responses):
    """Iterates through server responses, sends text to Gemini, and plays the result."""
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if result.is_final:
            transcript = result.alternatives[0].transcript.strip()
            if transcript:  # Only process if there's actual text
                print(f"Transcript: {transcript}")
                
                # Get response from Gemini AI
                gemini_response = get_gemini_response(transcript)
                print(f"Gemini Response: {gemini_response}")
                
                # Synthesize and play the Gemini response
                synthesize_and_play_audio(gemini_response)
                
                # Stop if the keyword "stop" is detected
                if 'stop' in transcript.lower():
                    print("Stopping transcription...")
                    break

def main():
    # Set up the PyAudio stream
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    # Configure recognition settings
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code='en-US',
        enable_automatic_punctuation=True
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    try:
        # Start streaming audio to the Google Cloud Speech API
        audio_generator = generate_audio_chunks(stream)
        responses = client.streaming_recognize(config=streaming_config, requests=audio_generator)

        # Print the results and synthesize responses
        listen_print_loop(responses)

    except KeyboardInterrupt:
        print("Transcription stopped by user.")

    finally:
        # Make sure the stream is closed after we're done
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

if __name__ == "__main__":
    main()