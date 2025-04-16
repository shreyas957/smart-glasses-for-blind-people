import cv2
import boto3
import pygame
import os
import time


textract_client = boto3.client('textract')
polly_client = boto3.client('polly')    

def play_audio(audio_file_path):
    # Initialize pygame for audio playback
    pygame.mixer.init()

    # Play the synthesized audio using pygame mixer
    print("Starting audio")
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)  # pause for 100 milliseconds

def process_and_play_audio():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from the webcam.")
        exit()

    # Convert frame to JPEG image bytes
    _, image_bytes = cv2.imencode('.jpg', frame)
    cap.release()

    # Detect text in the captured image using AWS Textract
    response = textract_client.detect_document_text(Document={'Bytes': image_bytes.tobytes()})

    # Extract recognized text from the response
    detected_texts = []
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE' and 'Text' in item:
            detected_texts.append(item['Text'])

    print(detected_texts)

    # Concatenate all detected text into a single string
    text_to_speak = ' '.join(detected_texts)

    # Synthesize speech using AWS Polly
    response = polly_client.synthesize_speech(
        Engine='generative',
        Text=text_to_speak,
        OutputFormat='mp3',
        VoiceId='Kajal'
    )

    # Save the audio stream to a temporary file
    audio_file_path = 'output.mp3'
    with open(audio_file_path, 'wb') as f:
        f.write(response['AudioStream'].read())

    # Call the function to play the audio
    play_audio(audio_file_path)

    # Clean up: Delete the temporary audio file
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

# Call the function to process and play audio
process_and_play_audio()

