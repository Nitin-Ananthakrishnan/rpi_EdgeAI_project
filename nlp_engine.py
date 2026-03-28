import os
import time
import services  # Import your services.py for the Weather API

class NLPEngine:
    def __init__(self):
        # State Management
        self.previous_sign = None
        self.last_spoken_time = 0
        
        # Engineering Constraint: Cooldown
        # Prevents the system from speaking too many times in a row
        self.cooldown_seconds = 4.0 

        # --- KNOWLEDGE BASE: LEVEL 2 (Single Signs) ---
        self.semantic_dictionary = {
            "Background": "",
            "Hello": "Hello there! I am using an AI assistant.",
            "Yes": "Yes, I agree.",
            "Thumbsup": "I am doing good, everything is okay.",
            "Pointing": "I would like that item, please.",
            "Raised": "I have a question or I need attention.",
            "Pinch": "Please adjust this or make it smaller.",
            "Call": "I need to make a phone call.",
            "Peace": "I am at peace, no problems here.",
            "L": "I need emergency assistance immediately."
        }

        # --- KNOWLEDGE BASE: LEVEL 3 (Contextual Sequences) ---
        self.context_dictionary = {
            # Pointing Interactions
            ("POINT", "YES"): "Yes, that is exactly the one I want.",
            ("POINT", "HELP"): "I need some help with this specific item.",
            ("POINT", "THANKYOU"): "Thank you, I will take this one.",
            
            # Hello / Emergency Interactions
            ("HELLO", "HELP"): "Hello, this is an emergency! Please help me.",
            ("HELLO", "POINT"): "Hello, could you show me that item please?",
            
            # Help Interactions
            ("HELP", "YES"): "Yes, I definitely need some assistance.",
            ("HELP", "POINT"): "Please help me move or reach that.",
            ("HELP", "THANKYOU"): "Thank you for coming to help me.",
            
            # Clarification / Confirmation
            ("THUMBSUP", "YES"): "Yes, everything is absolutely perfect.",
            ("THANKYOU", "THUMBSUP"): "Thank you, I am doing great now."
        }

    def speak_text(self, text):
        """
        Finalized Asynchronous Audio Pipeline for Raspberry Pi Bookworm.
        """
        temp_wav = f"/tmp/speech_{int(time.time())}.wav"
    
        # We use 'pw-play' (Native PipeWire) as it is the most modern 
        # and reliable tool for Bluetooth on the new Raspberry Pi OS.
        command = f'flite -t "{text}" -o {temp_wav} && pw-play {temp_wav} && rm {temp_wav} &'
    
        os.system(command)

    def process_and_speak(self, current_sign):
        """
        The core NLP logic: Decides what to say based on Current Sign 
        and Stored Context (Previous Sign).
        """
        # 1. Ignore Background/Null class
        if current_sign == "Background" or current_sign not in self.semantic_dictionary:
            self.previous_sign = None
            return

        # 2. Apply Cooldown (Safety Guard)
        if (time.time() - self.last_spoken_time) < self.cooldown_seconds:
            return

        sentence_to_speak = ""

        # --- LEVEL 3: DYNAMIC API TRIGGER ---
        # Special case: HELLO followed by THUMBSUP triggers the Weather Briefing
        if self.previous_sign == "HELLO" and current_sign == "THUMBSUP":
            print("[NLP ENGINE]: Triggering External Service: Daily Briefing")
            sentence_to_speak = services.get_daily_briefing()
            self.previous_sign = None # Clear context after service execution
        
        # --- LEVEL 3: STATIC CONTEXTUAL SEQUENCES ---
        elif self.previous_sign is not None:
            sequence = (self.previous_sign, current_sign)
            if sequence in self.context_dictionary:
                sentence_to_speak = self.context_dictionary[sequence]
                self.previous_sign = None # Sequence completed, reset memory
        
        # --- LEVEL 2: FALLBACK TO SINGLE SIGN ---
        if sentence_to_speak == "":
            # If no sequence was matched, speak the standard sign meaning
            sentence_to_speak = self.semantic_dictionary.get(current_sign, "")
            # Update memory for the NEXT gesture
            self.previous_sign = current_sign 

        # 3. Final Output Execution
        if sentence_to_speak != "":
            print(f"[AI RAW]: {current_sign} | [NLP SENTENCE]: {sentence_to_speak}")
            
            # Send to Audio Engine
            self.speak_text(sentence_to_speak)
            
            # Update last spoken timestamp
            self.last_spoken_time = time.time()

# --- STANDALONE TEST MODE ---
# You can run 'python3 nlp_engine.py' to test without the camera
if __name__ == "__main__":
    nlp = NLPEngine()
    print("NLP Engine Test Mode (Type 'QUIT' to stop)")
    while True:
        test_input = input("Enter Simulated Sign: ").upper()
        if test_input == "QUIT": break
        nlp.process_and_speak(test_input)
