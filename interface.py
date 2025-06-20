from model_loader import ModelLoader
from chat_memory import ChatMemory

class ChatInterface:
    def __init__(self):
        self.model = ModelLoader()
        self.memory = ChatMemory(window_size=3)
        
    def start(self):
        print("=== Local Chatbot ===")
        print("Loading...")
        
        if not self.model.load_model():
            print("Failed to load model!")
            return
        
        print("Ready! Type /exit to quit")
        print("-" * 30)
        
        while True:
            user_input = input("User: ").strip()
            
            if user_input.lower() == "/exit":
                print("Exiting chatbot. Goodbye!")
                break
                
            if not user_input:
                continue
                
            chat_history_ids = self.memory.get_chat_history_ids()
            response, new_chat_history_ids = self.model.generate_response(user_input, chat_history_ids)
            
            print(f"Bot: {response}")
            
            self.memory.add_conversation(user_input, response, new_chat_history_ids)
