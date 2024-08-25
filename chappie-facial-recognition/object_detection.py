import cv2
import boto3
import pygame
import os


rekognition_client = boto3.client('rekognition')
polly_client = boto3.client('polly')

def play_audio(audio_file_path):
    # Initialize Pygame for audio playback
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.quit()

def process_and_play_audio():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    try:
        # Capture frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame from the webcam.")
            exit()

        # Convert frame to JPEG image bytes
        _, image_bytes = cv2.imencode('.jpg', frame)

        # Detect objects, scenes, and text in the image using AWS Rekognition
        response = rekognition_client.detect_labels(Image={'Bytes': image_bytes.tobytes()})
        labels = [label['Name'] for label in response['Labels']]

        # Create a descriptive text based on detected labels
        scene_description = f"This scene contains: {', '.join(labels)}"

        # Synthesize speech from the descriptive text
        response = polly_client.synthesize_speech(
            Text=scene_description,
            OutputFormat='mp3',
            VoiceId='Joanna'
        )

        # Save the synthesized speech to a temporary file
        audio_file_path = 'scene_description.mp3'
        with open(audio_file_path, 'wb') as f:
            f.write(response['AudioStream'].read())

        # Play the synthesized audio using Pygame mixer
        play_audio(audio_file_path)

    finally:
        # Release webcam and cleanup Pygame resources
        cap.release()
        pygame.mixer.quit()

        # Delete the temporary audio file
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

# Call the function to process and play audio (Only if the file is going to run)
process_and_play_audio()
