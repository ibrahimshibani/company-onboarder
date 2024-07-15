import os
from dotenv import load_dotenv
import yaml
from langchain_openai import OpenAI
from utils.rag_utils import start_template, get_company_data, preprocess_input

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

class Chatbot:
    def __init__(self):
        self.llm = OpenAI(api_key=api_key, model="gpt-3.5-turbo-instruct")
        self.context = get_company_data()
        self.conversation_history = []
        self.conv_length = 0


    def get_response(self, user_input):
        self.conversation_history.append(f"User: {user_input}")
        conversation_context = "\n".join(self.conversation_history)
        preprocessed_input = preprocess_input(conversation_context)
        prompt = start_template.format(context=self.context, question=conversation_context)
        try:
            response = self.llm.invoke(prompt)
            self.conversation_history.append(response)
            self.conv_length += 1
        except Exception as e:
            response = f"An error occurred: {e}"
            print("ERROR", response)
        return response

def main():
    chatbot = Chatbot()  
    print("Chatbot: Hello I am your Jedox HR Assistant, what can I help you with today?")  
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Chatbot: Goodbye!")
                break
            response = chatbot.get_response(user_input)
            print(response)
        except KeyboardInterrupt:
            print("\nChatbot: Goodbye!")
            break

if __name__ == "__main__":
    main()
