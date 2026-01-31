import pyttsx3

class TTS:
    def __init__(self):
        pass
    
    def speak(self, text):
        if text and text.strip():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 190)
                engine.setProperty('volume', 0.9)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Speech failed: {e}")

                