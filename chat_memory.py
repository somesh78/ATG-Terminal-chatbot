class ChatMemory:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.history = []  
        self.chat_history_ids = None  
        
    def add_conversation(self, user_input, bot_response, chat_history_ids=None):
        self.history.append((user_input, bot_response))
        
        if len(self.history) > self.window_size:
            self.history.pop(0) 
        
        self.chat_history_ids = chat_history_ids
    
    def get_context(self):
        """Get conversation context as string"""
        if not self.history:
            return ""
        
        context = ""
        for user_msg, bot_msg in self.history[-2:]:
            context += f"User: {user_msg}\nBot: {bot_msg}\n"
        
        return context
    
    def get_chat_history_ids(self):
        return self.chat_history_ids
    
    def clear(self):
        self.history = []
        self.chat_history_ids = None