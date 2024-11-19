import os
import json
import threading
from datetime import datetime
import sounddevice as sd
from playsound import playsound
from melo.api import TTS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from speech_to_text import record_audio, transcribe_audio

# Original master/template
with open("llm_template.txt", "r") as f:
    master = f.read()

# Constants and default settings
MAX_HISTORY_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB
DEFAULT_SETTINGS = {
    'text_mode': True,
    'language': 'EN',  # Default language for TTS
    'history_size': 5,  # Default to keep last 5 history items
    'model_name': 'llama3.1',  # Default model name
}
SETTINGS_FILE = 'assistant_settings.json'

# Load settings from a file or use default settings
def load_settings():
    """Load settings from a JSON file, if it exists."""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    else:
        # If no settings file exists, use the default settings
        return DEFAULT_SETTINGS

# Save settings to a JSON file
def save_settings(settings):
    """Save settings to a JSON file."""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)

# Print the settings menu
def print_settings_menu(settings):
    """Print the current settings menu."""
    print("\n\033[36m" + "_"*40 + "\033[0m")  # Cyan for separator
    print("\033[34m--- Settings Menu ---\033[0m")
    print("\033[33m1. Text Mode: {}\033[0m".format("Enabled" if settings['text_mode'] else "Disabled"))
    print("\033[33m2. Language: {}\033[0m".format(settings['language']))
    print("\033[33m3. Chat History Size: {}\033[0m".format(settings['history_size']))
    print("\033[33m4. Model: {}\033[0m".format(settings['model_name']))
    print("\033[33m5. Remove History Items\033[0m")
    print("\033[33m6. Exit Settings\033[0m")

# Handle the user's choice in the settings menu
def handle_settings_choice(settings, client):
    """Handle the user's choice in the settings menu."""
    while True:
        print_settings_menu(settings)
        user_choice = input("\033[32mPlease select an option (1-6): \033[0m")
        
        if user_choice == '1':
            settings['text_mode'] = not settings['text_mode']
            print("\033[33mText Mode has been {}\033[0m".format("Enabled" if settings['text_mode'] else "Disabled"))
        elif user_choice == '2':
            new_language = input("\033[32mEnter language code (e.g., EN, FR, etc.): \033[0m").strip()
            settings['language'] = new_language
            print("\033[33mLanguage set to: {}\033[0m".format(new_language))
        elif user_choice == '3':
            try:
                new_history_size = int(input("\033[32mEnter number of history items to keep: \033[0m"))
                settings['history_size'] = new_history_size
                print("\033[33mHistory size set to: {}\033[0m".format(new_history_size))
            except ValueError:
                print("\033[31mInvalid input. Please enter a number.\033[0m")
        elif user_choice == '4':
            new_model = input("\033[32mEnter model name (e.g., llama3.1): \033[0m").strip()
            settings['model_name'] = new_model
            print("\033[33mModel set to: {}\033[0m".format(new_model))
        elif user_choice == '5':
            # Option to remove history items
            try:
                num_to_remove = int(input("\033[32mEnter the number of history items to remove from the end: \033[0m"))
                client.remove_history_items(num_to_remove)
                print(f"\033[33mRemoved {num_to_remove} items from the history.\033[0m")
            except ValueError:
                print("\033[31mInvalid input. Please enter a valid number.\033[0m")
        elif user_choice == '6':
            print("\033[34mExiting settings...\033[0m")
            save_settings(settings)  # Save settings before exiting
            break
        else:
            print("\033[31mInvalid option. Please try again.\033[0m")

# Settings entry point
def enter_settings(client):
    """Start the settings menu and manage settings changes."""
    settings = load_settings()  # Load current settings
    handle_settings_choice(settings, client)  # Show the menu and allow the user to adjust settings

# AI Assistant main class
class ChatMLMessage(BaseModel):
    role: str
    content: str

class Client:
    def __init__(self, history: list[ChatMLMessage] = []):
        self.greet()
        self.history = history
        self.settings = load_settings()  # Load settings
        self.tts = TTS(language=self.settings['language'], device="cpu")
        self.text_mode = self.settings['text_mode']  # Initialize text_mode from settings

        # Ollama model for AI responses
        self.model_name = self.settings['model_name']
        self.model = OllamaLLM(model=self.model_name)
        self.prompt = ChatPromptTemplate.from_template(master)
        self.chain = self.prompt | self.model

        # Start the conversation loop in a separate thread
        t = threading.Thread(target=self.conversation_loop)
        t.start()

    def greet(self):
        print("\n\033[36m--- Initializing AI Assistant... Please wait. ---\033[0m")  # Cyan

    def addToHistory(self, content: str, role: str):
        """Add the user or assistant message to history and print it."""
        if role == "user":
            print(f"\n\033[32m- User: {content}\033[0m")  # Green for user
        else:
            print(f"\n\033[33m- Assistant: {content}\033[0m")  # Yellow for assistant
        
        # Append the new message to history
        self.history.append(ChatMLMessage(content=content, role=role))

        # Keep only the last 'history_size' messages in history
        if len(self.history) > self.settings['history_size'] * 2:  # Keep twice the amount of messages
            self.history = self.history[-self.settings['history_size'] * 2:]

        # Now save the updated history to a JSON file
        self.save_history_to_json()

    def save_history_to_json(self):
        """Save the chat history to the JSON file, managing file size."""
        history_dict = [msg.dict() for msg in self.history]
        data = {"history": history_dict}

        # Check if the file exists and is too large before saving
        if os.path.exists('chat_history.json') and os.path.getsize('chat_history.json') > MAX_HISTORY_SIZE:
            self.handle_large_file()

        # Save the (reduced or cleared) history to the file
        with open("chat_history.json", "w") as file:
            json.dump(data, file, indent=4)
            print("\033[32mHistory saved successfully.\033[0m")  # Green

    def remove_history_items(self, num_items):
        """Remove a certain number of items from the end of the history."""
        if num_items <= len(self.history):
            self.history = self.history[:-num_items]
            self.save_history_to_json()
            print(f"\033[33mRemoved {num_items} history items.\033[0m")
        else:
            print("\033[31mCannot remove more items than are in the history.\033[0m")

    def handle_large_file(self):
        """Handle situations where the file is too large."""
        print("\033[31mHistory file is too large. Would you like to reduce it?\033[0m")  # Red
        print("\033[31mOptions to reduce:\033[0m")  # Red
        print("\033[31m1. Keep only the last 1/3 of the history.\033[0m")  # Red
        print("\033[31m2. Keep only the last 1/2 of the history.\033[0m")  # Red
        print("\033[31m3. Keep only the last 1/4 of the history.\033[0m")  # Red
        print("\033[31m4. Remove all history.\033[0m")  # Red

        # Get user input to decide how much history to keep
        user_input = input("\033[31mEnter '1', '2', '3', or '4' to select your option: \033[0m")  # Red

        # Decide how much to reduce based on user input
        if user_input == '1':
            # Keep the last 1/3
            print("\033[33mReducing history size to the last 1/3...\033[0m")  # Yellow
            new_size = len(self.history) // 3
            self.history = self.history[-new_size:]
        elif user_input == '2':
            # Keep the last 1/2
            print("\033[33mReducing history size to the last 1/2...\033[0m")  # Yellow
            new_size = len(self.history) // 2
            self.history = self.history[-new_size:]
        elif user_input == '3':
            # Keep the last 1/4
            print("\033[33mReducing history size to the last 1/4...\033[0m")  # Yellow
            new_size = len(self.history) // 4
            self.history = self.history[-new_size:]
        elif user_input == '4':
            # Remove all history
            print("\033[33mRemoving all history...\033[0m")  # Yellow
            self.history = []  # Clear history
        else:
            print("\033[33mInvalid option. Keeping full history.\033[0m")  # Yellow

    def print_separator(self):
        """Print a visual separator for clarity."""
        print("\n\033[36m" + "_"*40 + "\033[0m")  # Cyan for separator

    def print_chat_history(self):
        """Print the chat history in blue."""
        print("\n\033[34m--- Chat History ---\033[0m\n")  # Blue for history
        history_str = "\n".join([f"\033[32m{msg.role.capitalize()}: {msg.content}\033[0m" for msg in self.history])
        print(history_str if history_str else "\033[33mNo conversation history yet.\033[0m")  # Yellow for empty history

    def handle_mode_switch(self, user_input):
        """Switch between text and voice modes based on user input."""
        normalized_input = user_input.replace(" ", "").lower()  # Remove spaces and convert to lowercase

        if 'settings' in normalized_input or 'chatsettings' in normalized_input:
            enter_settings(self)  # This will take the user to the settings menu
            return True

        # Check for 'textmode' switch when currently in voice mode
        if ('textmode' in normalized_input or 'switchtotextmode' in normalized_input) and not self.text_mode:
            self.text_mode = True
            print("\033[33m\nSwitched to Text Mode. You can now type your input.\033[0m")  # Yellow
            return True

        # Check for 'voicemode' switch when currently in text mode
        if ('voicemode' in normalized_input or 'switchtovoicemode' in normalized_input) and self.text_mode:
            self.text_mode = False
            print("\033[33m\nSwitched to Voice Mode. Speak your input.\033[0m")  # Yellow
            return True
        
        return False

    def conversation_loop(self):
        while True:
            try:
                self.print_separator()
                self.print_chat_history()

                if not self.text_mode:
                    # Capture audio input using Whisper when in voice mode
                    self.print_separator()
                    user_input = self.listen_to_audio()
                    print("\n\033[36m[Voice Mode] Listening for your input...\033[0m\n")  # Cyan for voice input
                else:
                    self.print_separator()
                    # Capture text input when in text mode
                    user_input = input("\n\033[36mPlease enter your input: \033[0m")  # Cyan for text input

                # Handle switching between modes based on user input
                if self.handle_mode_switch(user_input):
                    continue

                if user_input == "":
                    continue  # Skip empty inputs
                
                self.addToHistory(user_input, "user")

                current_time = datetime.now().strftime("%I:%M %p")
                current_date = datetime.now().strftime("%Y-%m-%d")

                history = self.getHistoryAsString()
                response = self.chain.invoke({
                    "input": user_input,
                    "history": history,
                    "time": current_time,
                    "date": current_date
                })

                assistant_response = response
                self.addToHistory(assistant_response, "assistant")
                self.speak(assistant_response)

            except Exception as e:
                self.handle_error(f"An error occurred during the conversation: {e}")

    def getHistoryAsString(self) -> str:
        """Return the last 5 messages formatted as a string."""
        history = self.history[-10:]  # The most recent 5 exchanges (10 messages total)
        return "\n".join([f"<|{msg.role}|>{msg.content}<|end|>" for msg in history])

    def listen_to_audio(self):
        try:
            audio = record_audio()
            transcription = transcribe_audio(audio)
            return transcription.lower()
        except Exception as e:
            print(f"\033[31m- An error has occurred: {e}\033[0m")  # Red for errors

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

        except Exception as e:
            self.handle_error(f"An error has occurred: {e}")

    def handle_error(self, error_message):
        """Handles errors in a clean and consistent way."""
        print(f"\n\033[31m[Error] {error_message}\033[0m")  # Red for errors

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
