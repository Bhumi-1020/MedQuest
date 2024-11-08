import pyaudio
import google.generativeai as genai
import os
from google.cloud import speech
from google.cloud import texttospeech
import tempfile
import pygame
import random
import pandas as pd
import re 
from evaluation import evaluate_diagnosis, provide_feedback

# Set environment variable for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(), 'keySecondTTS.json')

# Configure API key for Google Generative AI
genai.configure(api_key='AIzaSyCXKWaSlWsgALM-D89HPfn4aP2h3sewcnA')

# Set up Google Cloud Speech clients
client = speech.SpeechClient.from_service_account_file('keySecondSTT.json')

# Audio recording parameters
RATE = 44100
CHUNK = int(RATE / 10)  # 100ms

# Load the Medical Transcriptions dataset
df = pd.read_csv('dataset/mtsamples.csv')  

def determine_patient_gender(patient_info):
    """Determine the patient's gender based on the available information."""
    text_to_search = (patient_info['transcription'] + ' ' + 
                      patient_info['description'] + ' ' + 
                      patient_info['keywords']).lower()

    # Use regular expressions to search for gender-specific keywords
    gender_patterns = {
    'female': r'\b(she|her|hers|woman|female|girl|breast|ovarian cancer|cervical cancer|endometriosis|polycystic ovary syndrome|uterus|vagina|menstruation|menstrual|pregnancy|pregnant|maternity|ovulation|estrogen|gynecologist|gynaecologist|mammogram|fallopian)\b',
    'male': r'\b(he|him|his|man|male|boy|testicular|testicle|prostate|erectile dysfunction|testosterone|androgen|scrotum|sperm|vasectomy|testes|seminal|ejaculation|urologist|penis|erection)\b'
}

    for gender, pattern in gender_patterns.items():
        if re.search(pattern, text_to_search):
            return gender

    return 'unknown'

def select_random_patient():
    """Select a random patient from the dataset."""
    patient = df.sample(n=1).iloc[0]
    patient_info = {
        'transcription': patient['transcription'],
        'medical_specialty': patient['medical_specialty'],
        'sample_name': patient['sample_name'],
        'description': patient['description'],
        'keywords': patient['keywords']
    }
    patient_info['gender'] = determine_patient_gender(patient_info)
    return patient_info

current_patient = select_random_patient()

def generate_audio_chunks(stream):
    """Generator that yields audio chunks from the microphone."""
    while True:
        data = stream.read(CHUNK)
        if not data:
            break
        yield speech.StreamingRecognizeRequest(audio_content=data)

def get_virtual_patient_response(text):
    """Send text to Gemini AI and get the response as a virtual patient."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""You are a virtual patient based on the following medical transcription:

    Medical Specialty: {current_patient['medical_specialty']}
    Sample Name: {current_patient['sample_name']}
    Description: {current_patient['description']}
    Keywords: {current_patient['keywords']}

    Here's an excerpt from your medical transcription:
    {current_patient['transcription'][:500]}...

    Based on this information, respond to the following query from a healthcare provider:

    {text}

    Remember to stay in character and provide responses consistent with the medical transcription. If asked about details not provided in the transcription, improvise in a manner consistent with the given information."""
    
    response = model.generate_content(prompt)
    cleaned_response = clean_up_text(response.text)
    return cleaned_response

def clean_up_text(text):
    """Clean up the response text to make it sound more natural."""
    text = text.replace("As an AI", "")
    text = text.replace("As a virtual patient", "")
    text = text.replace("*", "")
    text = text.replace("", "")
    text = text.replace("!!", "!")
    return text.strip()

def synthesize_and_play_audio(text, gender):
    """Synthesize text to speech and play the audio."""
    # Initialize the Text-to-Speech client
    tts_client = texttospeech.TextToSpeechClient()

    # Set the text input
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Select a random voice
    # voice = texttospeech.VoiceSelectionParams(
    #     language_code="en-US",
    #     name=random.choice(['en-US-Wavenet-A', 'en-US-Wavenet-B', 'en-US-Wavenet-C', 'en-US-Wavenet-D', 'en-US-Wavenet-E', 'en-US-Wavenet-F'])
    # )

    # Select a voice based on gender
    if gender == 'female':
        voice_name = random.choice(['en-US-Wavenet-C', 'en-US-Wavenet-E', 'en-US-Wavenet-F'])
    elif gender == 'male':
        voice_name = random.choice(['en-US-Wavenet-A', 'en-US-Wavenet-B', 'en-US-Wavenet-D'])
    else:
        voice_name = random.choice(['en-US-Wavenet-A', 'en-US-Wavenet-B', 'en-US-Wavenet-C', 'en-US-Wavenet-D', 'en-US-Wavenet-E', 'en-US-Wavenet-F'])

    voice = texttospeech.VoiceSelectionParams(
        language_code = "en-US",
        name = voice_name
    )


    # Configure the audio output
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=0.9,
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
    """Iterates through server responses, sends text to virtual patient, and plays the result."""
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if result.is_final:
            transcript = result.alternatives[0].transcript.strip()
            if transcript:  # Only process if there's actual text
                print(f"Healthcare Provider: {transcript}")
                
                # Get response from virtual patient
                patient_response = get_virtual_patient_response(transcript)
                print(f"Virtual Patient: {patient_response}")
                
                # Synthesize and play the virtual patient's response
                synthesize_and_play_audio(patient_response, current_patient['gender'])
                
                # Stop if the keyword "end session" is detected
                if 'end session' in transcript.lower():
                    print("Ending virtual patient session...")
                    break

def main():
    print(f"Starting virtual patient session.")
    print(f"Medical Specialty: {current_patient['medical_specialty']}")
    print(f"Name: {current_patient['sample_name']}")
    print(f"Description: {current_patient['description']}")
    print(f"Keywords: {current_patient['keywords']}")
    print(f"Gender: {current_patient['gender']}")
    
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

    full_transcript = ""

    try:
        # Start streaming audio to the Google Cloud Speech API
        audio_generator = generate_audio_chunks(stream)
        responses = client.streaming_recognize(config=streaming_config, requests=audio_generator)

        # Print the results and synthesize responses
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if result.is_final:
                transcript = result.alternatives[0].transcript.strip()
                if transcript:
                    print(f"Healthcare Provider: {transcript}")
                    full_transcript += f"Healthcare Provider: {transcript}\n"
                    
                    # Get response from virtual patient
                    patient_response = get_virtual_patient_response(transcript)
                    print(f"Virtual Patient: {patient_response}")
                    full_transcript += f"Virtual Patient: {patient_response}\n"
                    
                    # Synthesize and play the virtual patient's response
                    synthesize_and_play_audio(patient_response, current_patient['gender'])
                    
                    # Stop if the keyword "end session" is detected
                    if 'end session' in transcript.lower():
                        print("Ending virtual patient session...")
                        break

    except KeyboardInterrupt:
        print("Virtual patient session stopped by user.")

    finally:
        # Make sure the stream is closed after we're done
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

    # Automatically send the full transcript for evaluation
    evaluation_results = evaluate_diagnosis(full_transcript, current_patient)
    
    # Provide feedback
    feedback = provide_feedback(evaluation_results)
    print(feedback)

if __name__ == "__main__":
    main()
