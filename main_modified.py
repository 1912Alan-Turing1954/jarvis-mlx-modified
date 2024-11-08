import time
from datetime import datetime
import threading
from playsound import playsound
from melo.api import TTS
from pydantic import BaseModel
import ollama  # Ollama for AI chat
import librosa
import sounddevice as sd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# master = "You are a helpful assistant designed to run offline with decent latency, you are open source. Answer the following input from the user in no more than three sentences. Address them as Sir at all times. Only respond with the dialogue, nothing else."
master = "You are Edith, a helpful assistant designed to run offline with decent latency, you are open source. You are concise, direct, supportive, intelligent, and friendly, with a caring yet relaxed tone. You are loyal, resourceful, and always ready to lend a hand in a down-to-earth way. Answer the following input from the user in no more than two sentences. Address them as sir or logan. Only respond with the dialogue, nothing else: input={input}, chat_history={history}, time={time}"

class ChatMLMessage(BaseModel):
    role: str
    content: str

class Client:
    def __init__(self, history: list[ChatMLMessage] = []):
        self.greet()
        self.history = history
        self.tts = TTS(language="EN", device="cpu")
        self.current_time = datetime.now().strftime("%Y-%m-%d %I:%M %p")

        # Ollama model for AI responses
        self.model_name = "llama3.1"
        self.model = OllamaLLM(model=self.model_name)
        self.prompt = ChatPromptTemplate.from_template(master)
        self.chain = self.prompt | self.model

        # Start a thread to handle the conversation loop
        t = threading.Thread(target=self.conversation_loop)
        t.start()

    def greet(self):
        print()
        print("\033[36mWelcome to JARVIS-MLX\n\nFollow @huwprosser_ on X for updates\033[0m")
        print()

    def addToHistory(self, content: str, role: str):
        """Add the user or assistant message to history and print it."""
        if role == "user":
            print(f"\033[32m{content}\033[0m")
        else:
            print(f"\033[33m{content}\033[0m")
        
        # If it's a user message, append the master context
        if role == "user":
            content = f"{master}\n\n{content}"
        self.history.append(ChatMLMessage(content=content, role=role))

    def getHistoryAsString(self):
        """Return the chat history as a formatted string."""
        final_str = ""
        for message in self.history:
            final_str += f"<|{message.role}|>{message.content}<|end|>\n"
        return final_str

    def conversation_loop(self):
        """Main loop for handling user input and processing AI responses."""
        while True:
            user_input = input("\033[36mPlease enter your input: \033[0m")  # Simulating user input
            if not user_input.strip():
                continue  # Skip if no input is provided

            self.addToHistory(user_input, "user")  # Add user input to history

            # Get the conversation history as string
            history = self.getHistoryAsString()

            response = self.chain.invoke({
                "input": user_input,
                "history": history,
                "time": self.current_time
            })

            # Extract response from Ollama's response
            assistant_response = response

            self.addToHistory(assistant_response, "assistant")  # Add assistant's response to history

            # Use TTS to speak out the assistant's response
            self.speak(assistant_response)

    def speak(self, text):
        """Use TTS (Text-to-Speech) to speak the response."""
        speaker_ids = self.tts.hps.data.spk2id

        print(speaker_ids)

        # Generate audio file from text
        audio_data = self.tts.tts_to_file(
            text,
            speaker_ids["EN-Default"],
            speed=0.94,
            quiet=True,
            sdp_ratio=0.5,
            noise_scale=1,
            noise_scale_w=0.8,
        )

        # Play the raw audio data using sounddevice (no need to use librosa)
        sd.play(audio_data, 44100, blocking=True)  # Play at 44.1 kHz sample rate
        print(f"\033[36m{text}\033[0m")
        
        time.sleep(1)  # Delay to simulate the pause between responses
if __name__ == "__main__":
    jc = Client(history=[])
