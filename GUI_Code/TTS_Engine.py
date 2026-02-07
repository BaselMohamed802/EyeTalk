import pyttsx3

class TTS:
    def __init__(self):
        self.engine = pyttsx3.init(driverName='sapi5')  # Windows SAPI
        
        # Set properties
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        
        # Find voices
        voices = self.engine.getProperty('voices')
        
        # Print available voices
        print("Available voices:")
        for voice in voices:
            print(f"  - {voice.name}")
        
        # Try to use Arabic voice if installed
        for voice in voices:
            if 'arabic' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
    
    def speak(self, text):
        if text and text.strip():
            self.engine.say(text)
            self.engine.runAndWait() 