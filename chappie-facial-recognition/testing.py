import cv2 as cv
import pygame
import argparse
import boto3
import time
import os
import speech_recognition as sr

polly_client = boto3.client('polly')
pygame.mixer.init()

response = polly_client.synthesize_speech(
    Engine='generative',
    Text=f'Welcome to smart glasses, happy to help you', 
    OutputFormat='mp3', 
    VoiceId='Kajal'
)
with open('beep.mp3', 'wb') as f:
    f.write(response['AudioStream'].read())
    pygame.mixer.music.load('beep.mp3')
    pygame.mixer.music.play()
