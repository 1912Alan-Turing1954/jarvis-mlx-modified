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
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

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
        t.daemon = True  # Daemonize the thread so it doesn't block program exit
        t.start()

    def greet(self):
        print(Fore.CYAN + "\nInitializing AI Assistant... Please wait.\n")

    def addToHistory(self, content: str, role: str):
        """Add the user or assistant message to history and print it."""
        if role == "user":
            print(Fore.GREEN + f"- User: {content}")
        else:
            print(Fore.YELLOW + f"- Assistant: {content}")

        # Append the new message to history
        self.history.append(ChatMLMessage(content=content, role=role))

        # Keep only the last 10 messages in history (5 pairs of user-assistant)
        if len(self.history) > 10:  # 5 pairs of user-assistant = 10 total messages
            self.history = self.history[-10:]

        # Save the updated history to a JSON file
        self.save_history_to_json()

    def save_history_to_json(self):
        """Save the chat history to the JSON file, managing file size."""
        history_dict = [msg.dict() for msg in self.history]
        data = {"history": history_dict}

        with open("chat_history.json", "w") as file:
            json.dump(data, file, indent=4)

    def getHistoryAsString(self) -> str:
        """Return the last 5 messages formatted as a string."""
        return "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in self.history[-10:]])

    def conversation_loop(self):
        while True:
            try:
                # Display the chat history
                print(Fore.BLUE + "\n[History]\n" + Fore.WHITE + self.getHistoryAsString())

                # Handle input
                if not self.text_mode:
                    # Capture audio input using Whisper when in voice mode
                    user_input = self.listen_to_audio()
                    print(Fore.CYAN + "[Listening for your voice input...]")  # Cyan for voice input
                else:
                    # Capture text input when in text mode
                    user_input = input(Fore.GREEN + "\n[Please enter your input]: ")

                # Handle switching between modes based on user input
                normalized_input = user_input.replace(" ", "").lower()  # Normalize input by removing spaces and converting to lowercase

                if ('textmode' in normalized_input or 'switchtotextmode' in normalized_input) and not self.text_mode:
                    # Switch to text mode if user asks for it and we're in voice mode
                    self.text_mode = True
                    print(Fore.YELLOW + "[Switched to Text Mode. Please type your input.]")
                    continue

                elif ('voicemode' in normalized_input or 'switchtovoicemode' in normalized_input) and self.text_mode:
                    # Switch to voice mode if user asks for it and we're in text mode
                    self.text_mode = False
                    print(Fore.YELLOW + "[Switched to Voice Mode. Please speak your input.]")
                    continue

                elif user_input == '':
                    continue

                # Process user input and add to history
                playsound("beep.mp3")
                self.addToHistory(user_input, "user")

                # Prepare the conversation context
                history = self.getHistoryAsString()
                response = self.chain.invoke({
                    "input": user_input,
                    "history": history,
                    "time": self.current_time,
                    "date": self.current_date
                })

                assistant_response = response
                self.addToHistory(assistant_response, "assistant")

                # Speak the assistant response
                self.speak(assistant_response)

            except Exception as e:
                print(Fore.RED + f"[Error] An error has occurred: {e}")

    def listen_to_audio(self):
        try:
            audio = record_audio()
            transcription = transcribe_audio(audio)
            return transcription.lower()
        except Exception as e:
            print(Fore.RED + f"[Error] An error has occurred while listening: {e}")
            return ""

    def speak(self, text):
        try:
            """Use TTS (Text-to-Speech) to speak the response."""
            speaker_ids = self.tts.hps.data.spk2id
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
            sd.play(audio_data, 44100, blocking=True)  # Play at 44.1 kHz sample rate
            print(Fore.YELLOW + f"[Assistant] {text}")

        except Exception as e:
            print(Fore.RED + "[Error] An error occurred while speaking.")
            print(f"[Error] {e}")


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
