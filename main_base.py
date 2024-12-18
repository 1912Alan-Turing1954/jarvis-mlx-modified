import warnings

# Suppress specific BERT warning
warnings.filterwarnings("ignore")

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

# Original master/template
# master = "You are a helpful assistant designed to run offline with decent latency, you are open source. Answer the following input from the user in no more than three sentences. Address them as Sir at all times. Only respond with the dialogue, nothing else."

# In file master/template
# master = "You are Edith, a helpful assistant. You are concise, direct, supportive, intelligent, and friendly, with a caring yet relaxed and professional tone. You are loyal, resourceful, and always ready to lend a hand and help them. Answer the following input from the user in no more than two sentences. Avoid corny dialogue and slangs of any kind. You may ask questions if you need contenxt or if you are curious. Address them as sir or logan. Only respond with the casual dialogue, nothing else: input={input}, chat_history={history}, time={time}, date={date}"

# Text file template
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
        print("\n\033[36mInitializing AI Assistant... Please wait.\033[0m")
        print()

    def addToHistory(self, content: str, role: str):
        """Add the user or assistant message to history and print it."""
        if role == "user":
            print(f"\033[32m- {content}\033[0m")
        else:
            print(f"\033[33m- {content}\033[0m")
        
        # If it's a user message, append the master context
        if role == "user":
            content = f"{master}\n\n{content}"
        self.history.append(ChatMLMessage(content=content, role=role))

    def getHistoryAsString(self) -> str:
        """Return the chat history as a formatted string."""
        final_str = ""
        for message in self.history:
            final_str += f"<|{message.role}|>{message.content}<|end|>\n"
        return final_str

    def conversation_loop(self):
        """Main loop for handling user input and processing AI responses."""
        while True:
            try:
                print(f"\n\033[37m- Chat History \n{self.getHistoryAsString()}\033[0m")


                user_input = input("\n\033[36mPlease enter your input: \033[0m")  # Simulating user input
                if not user_input.strip():
                    continue  # Skip if no input is provided
                
                playsound("beep.mp3")

                self.addToHistory(user_input, "user")  # Add user input to history

                # Get the conversation history as string
                history = self.getHistoryAsString()

                response = self.chain.invoke({
                    "input": user_input,
                    "history": history,
                    "time": self.current_time,
                    "date": self.current_date
                })
                
                # Extract response from Ollama's response
                assistant_response = response

                self.addToHistory(assistant_response, "assistant")  # Add assistant's response to history

                # Use TTS to speak out the assistant's response
                self.speak(assistant_response)

            except Exception as e:
                print('- An error has ocurred -')
                print(e)


    def speak(self, text):
        try:
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

            playsound("beep.mp3")

            # Play the raw audio data using sounddevice (no need to use librosa)
            sd.play(audio_data, 44100, blocking=True)  # Play at 44.1 kHz sample rate
            print(f"\033[36m{text}\033[0m")
    
            time.sleep(1)  # Delay to simulate the pause between responses
        except Exception as e:
            print("An error has ocurred")
            print(e)

if __name__ == "__main__":
    jc = Client(history=[])