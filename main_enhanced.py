from datetime import datetime
import threading
import sounddevice as sd
from playsound import playsound
from melo.api import TTS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from speech_to_text import record_audio, transcribe_audio
import json
import os

MAX_HISTORY_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB


# Original master/template
with open("llm_template.txt", "r") as f:
    master = f.read()

class ChatMLMessage(BaseModel):
    role: str
    content: str

class Client:
    def __init__(self, history: list[ChatMLMessage] = []):
        self.greet()
        self.history = history
        self.tts = TTS(language="EN", device="cpu")
        self.current_time = datetime.now().strftime("%I:%M %p")
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.text_mode = True  # Initialize text_mode correctly

        # Ollama model for AI responses
        self.model_name = "llama3.1"
        self.model = OllamaLLM(model=self.model_name)
        self.prompt = ChatPromptTemplate.from_template(master)
        self.chain = self.prompt | self.model

        # Start the conversation loop in a separate thread
        t = threading.Thread(target=self.conversation_loop)
        t.start()

    def greet(self):
        print("\n\033[36mInitializing AI Assistant... Please wait.\033[0m")

    def addToHistory(self, content: str, role: str):
        """Add the user or assistant message to history and print it."""
        if role == "user":
            print(f"\033[32m- User: {content}\033[0m")
        else:
            print(f"\033[33m- Assistant: {content}\033[0m")
        
        # Append the new message to history
        self.history.append(ChatMLMessage(content=content, role=role))

        # Keep only the last 5 messages in history (5 messages = 2.5 exchanges, so 5 = 2 user + 2 assistant + 1 space)
        if len(self.history) > 10:  # 5 pairs of user-assistant = 10 total messages
            self.history = self.history[-10:]

        # Now save the updated history to a JSON file
        self.save_history_to_json()

    def save_history_to_json(self):
        """Save the chat history to the JSON file, managing file size."""
        history_dict = [msg.dict() for msg in self.history]
        data = {"history": history_dict}

        # Check if the file exists and is too large before saving
        if os.path.exists('chat_history.json') and os.path.getsize('chat_history.json') > MAX_HISTORY_SIZE:
            print("History file is too large. Would you like to reduce it?")
            print("Options to reduce:")
            print("1. Keep only the last 1/3 of the history.")
            print("2. Keep only the last 1/2 of the history.")
            print("3. Keep only the last 1/4 of the history.")
            user_input = input("Enter '1', '2', or '3' to select how much history to keep: ").strip()

            # Decide how much to reduce based on user input
            if user_input == '1':
                # Keep the last 1/3
                print("Reducing history size to the last 1/3...")
                new_size = len(self.history) // 3
                self.history = self.history[-new_size:]
            elif user_input == '2':
                # Keep the last 1/2
                print("Reducing history size to the last 1/2...")
                new_size = len(self.history) // 2
                self.history = self.history[-new_size:]
            elif user_input == '3':
                # Keep the last 1/4
                print("Reducing history size to the last 1/4...")
                new_size = len(self.history) // 4
                self.history = self.history[-new_size:]
            else:
                print("Invalid option. Keeping full history.")

        # Write to the file
        with open("chat_history.json", "w") as file:
            json.dump(data, file, indent=4)

    def getHistoryAsString(self) -> str:
        """Return the last 5 messages formatted as a string."""
        # Get the last 5 exchanges (user + assistant = 5 exchanges = 10 messages)
        history = self.history[-10:]  # The most recent 5 exchanges (10 messages total)
        
        # Convert the history to a formatted string
        return "\n".join([f"<|{msg.role}|>{msg.content}<|end|>" for msg in history])

    def conversation_loop(self):
        while True:
            try:
                print(f"\n\033[37m- Chat History \n{self.getHistoryAsString()}\033[0m")

                if not self.text_mode:
                    # Capture audio input using Whisper when in voice mode
                    user_input = self.listen_to_audio()
                    print("\033[36mListening for your voice input...\033[0m")  # Cyan for voice input
                else:
                    # Capture text input when in text mode
                    user_input = input("\n\033[32mPlease enter your input: \033[0m")  # Green for text input prompt

                # Handle switching between modes based on user input
                normalized_input = user_input.replace(" ", "").lower()  # Normalize input by removing spaces and converting to lowercase

                # Check if the user wants to switch to text mode
                if ('textmode' in normalized_input or 'switchtotextmode' in normalized_input) and not self.text_mode:
                    # Switch to text mode if user asks for it and we're in voice mode
                    self.text_mode = True
                    print("\033[33mSwitched to Text Mode. Please type your input.\033[0m")  # Yellow for switch message
                    # user_input = input("\n\033[32mPlease enter your input: \033[0m")  # Green for text input prompt
                    continue

                # Check if the user wants to switch to voice mode
                elif ('voicemode' in normalized_input or 'switchtovoicemode' in normalized_input) and self.text_mode:
                    # Switch to voice mode if user asks for it and we're in text mode
                    self.text_mode = False
                    print("\033[33mSwitched to Voice Mode. Please speak your input.\033[0m")  # Yellow for switch message
                    # user_input = self.listen_to_audio()
                    continue

                   # Handle empty input (do nothing and continue the loop)
                elif user_input == '':
                    continue
                
                else:
                    pass  # Explicitly doing nothing

                playsound("beep.mp3")
                self.addToHistory(user_input, "user")

                history = self.getHistoryAsString()
                response = self.chain.invoke({
                    "input": user_input,
                    "history": history,
                    "time": self.current_time,
                    "date": self.current_date
                })

                assistant_response = response

                self.addToHistory(assistant_response, "assistant")
                self.speak(assistant_response)

            except Exception as e:
                print(f"- An error has occurred: {e}")

    def listen_to_audio(self):
        try: 
            audio = record_audio()
            transcription = transcribe_audio(audio)
            return transcription.lower()
        except Exception as e:
            print(f"- An error has occurred: {e}")

    def speak(self, text):
        try:
            """Use TTS (Text-to-Speech) to speak the response."""
            speaker_ids = self.tts.hps.data.spk2id

            # Generate audio file from text (avoid repeated generation by saving to file)
            audio_data = self.tts.tts_to_file(
                text,
                speaker_ids["EN-Default"],
                speed=0.94,
                quiet=True,
                sdp_ratio=0.5,
                noise_scale=1,
                noise_scale_w=0.8,
            )

            playsound("beep.mp3")

            # Play the raw audio data using sounddevice (no need to use librosa)
            sd.play(audio_data, 44100, blocking=True)  # Play at 44.1 kHz sample rate
            print(f"\033[36m{text}\033[0m")

        except Exception as e:
            print("An error has occurred")
            print(e)

if __name__ == "__main__":
    # Load history from the JSON file, if it exists and is within size limit
    try:
        if os.path.exists('chat_history.json') and os.path.getsize('chat_history.json') <= MAX_HISTORY_SIZE:
            with open('chat_history.json', 'r') as history_file:
                history_data = json.load(history_file)
                history = [ChatMLMessage(**msg) for msg in history_data.get("history", [])]
        else:
            history = []

    except FileNotFoundError:
        history = []

    # Create a new Client instance with the loaded history
    jc = Client(history)