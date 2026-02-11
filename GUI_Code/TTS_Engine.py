import sys
import pyttsx3


class TTS:
    def __init__(self):
        self.engine = None

        # Pick a driver that exists on the current OS
        driver_candidates = []
        if sys.platform.startswith("win"):
            driver_candidates = ["sapi5"]
        elif sys.platform == "darwin":
            driver_candidates = ["nsss"]  # macOS
        else:
            # Linux (Raspberry Pi included)
            # pyttsx3 commonly supports: "espeak" (sometimes "espeak-ng")
            driver_candidates = ["espeak", "espeak-ng"]

        last_err = None
        for drv in driver_candidates:
            try:
                self.engine = pyttsx3.init(driverName=drv)
                break
            except Exception as e:
                last_err = e
                self.engine = None

        # Final fallback: let pyttsx3 try auto-detect
        if self.engine is None:
            try:
                self.engine = pyttsx3.init()
            except Exception as e:
                raise RuntimeError(
                    "TTS engine failed to initialize on this system. "
                    "On Raspberry Pi, install espeak-ng (see notes)."
                ) from e

        # Properties
        self.engine.setProperty("rate", 150)
        self.engine.setProperty("volume", 1.0)

        # Optional: choose an Arabic-ish voice if available
        try:
            voices = self.engine.getProperty("voices") or []
            for v in voices:
                name = (getattr(v, "name", "") or "").lower()
                vid = (getattr(v, "id", "") or "").lower()
                if "arab" in name or "ar_" in vid or "arabic" in name:
                    self.engine.setProperty("voice", v.id)
                    break
        except Exception:
            pass  # don't block the app if voice listing is weird on a platform

    def speak(self, text: str):
        if not self.engine:
            return
        if text and str(text).strip():
            self.engine.say(text)
            self.engine.runAndWait()
