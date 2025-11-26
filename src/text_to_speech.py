"""
Text-to-Speech Module
Provides text-to-speech functionality using pyttsx3.
"""

import threading
import queue


class TextToSpeech:
    """
    Text-to-speech engine using pyttsx3.
    Runs in a separate thread to avoid blocking the main application.
    """
    
    def __init__(self, rate=150, volume=1.0):
        """
        Initialize the TTS engine.
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        self.rate = rate
        self.volume = volume
        self.speech_queue = queue.Queue()
        self.is_running = True
        self.is_speaking = False
        
        # Start the speech thread
        self.speech_thread = threading.Thread(target=self._speech_loop, daemon=True)
        self.speech_thread.start()
    
    def _init_engine(self):
        """Initialize the pyttsx3 engine (must be done in the speech thread)."""
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)
        return engine
    
    def _speech_loop(self):
        """Main speech loop running in a separate thread."""
        engine = self._init_engine()
        
        while self.is_running:
            try:
                # Wait for text to speak (with timeout to check is_running)
                text = self.speech_queue.get(timeout=0.5)
                
                if text is None:
                    # Shutdown signal
                    break
                
                self.is_speaking = True
                engine.say(text)
                engine.runAndWait()
                self.is_speaking = False
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS Error: {e}")
                self.is_speaking = False
        
        engine.stop()
    
    def speak(self, text):
        """
        Queue text to be spoken.
        
        Args:
            text: The text to speak
        """
        if text and text.strip():
            self.speech_queue.put(text)
    
    def speak_letter(self, letter):
        """
        Speak a single letter.
        
        Args:
            letter: The letter to speak
        """
        self.speak(letter)
    
    def speak_word(self, word):
        """
        Speak a word.
        
        Args:
            word: The word to speak
        """
        self.speak(word)
    
    def clear_queue(self):
        """Clear any pending speech in the queue."""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
    
    def stop(self):
        """Stop the TTS engine and thread."""
        self.is_running = False
        self.speech_queue.put(None)  # Signal to exit
        self.speech_thread.join(timeout=2.0)
    
    def set_rate(self, rate):
        """
        Set the speech rate.
        
        Args:
            rate: Words per minute
        """
        self.rate = rate
    
    def set_volume(self, volume):
        """
        Set the volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.volume = max(0.0, min(1.0, volume))


class SpeechBuffer:
    """
    Buffers recognized letters and speaks words.
    """
    
    def __init__(self, tts_engine):
        """
        Initialize the speech buffer.
        
        Args:
            tts_engine: TextToSpeech instance
        """
        self.tts = tts_engine
        self.buffer = ""
    
    def add_letter(self, letter):
        """
        Add a letter to the buffer.
        
        Args:
            letter: The letter to add
        """
        self.buffer += letter
    
    def add_space(self):
        """Add a space to the buffer."""
        self.buffer += " "
    
    def backspace(self):
        """Remove the last character from the buffer."""
        if self.buffer:
            self.buffer = self.buffer[:-1]
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = ""
    
    def get_text(self):
        """Get the current buffer text."""
        return self.buffer
    
    def speak_buffer(self):
        """Speak the entire buffer."""
        if self.buffer.strip():
            self.tts.speak(self.buffer)
    
    def speak_last_word(self):
        """Speak the last word in the buffer."""
        words = self.buffer.strip().split()
        if words:
            self.tts.speak(words[-1])


if __name__ == "__main__":
    # Test the TTS module
    print("Testing Text-to-Speech module...")
    
    tts = TextToSpeech()
    
    # Test speaking
    tts.speak("Hello, I am the ASL Sign Language Translator")
    
    import time
    time.sleep(3)
    
    tts.speak("Testing letter pronunciation: A B C D E")
    
    time.sleep(4)
    
    # Test the buffer
    buffer = SpeechBuffer(tts)
    buffer.add_letter('H')
    buffer.add_letter('E')
    buffer.add_letter('L')
    buffer.add_letter('L')
    buffer.add_letter('O')
    
    print(f"Buffer contents: {buffer.get_text()}")
    buffer.speak_buffer()
    
    time.sleep(2)
    
    tts.stop()
    print("TTS test complete!")

