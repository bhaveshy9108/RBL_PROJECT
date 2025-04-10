from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the Groq model
groq_api_key = os.getenv("GROQ_API_KEY")
chat_model = ChatGroq(
api_key=groq_api_key,
model_name="llama-3.3-70b-versatile",
temperature=0.7,
)

# Create a template for mental health conversations
template = """
You are an empathetic and supportive mental health chatbot. Your role is to provide a safe space for users 
to discuss their feelings and concerns. While you're not a replacement for professional mental health care, 
you can offer support and general guidance.

Current conversation:
{history}
Human: {input}
AI Assistant: Let me support you with that."""

# Set up the prompt template
prompt = ChatPromptTemplate.from_template(template)

# Initialize conversation memory
memory = ConversationBufferMemory()

# Create the conversation chain
conversation = ConversationChain(
    llm=chat_model,
    prompt=prompt,
    memory=memory,
    verbose=True
)

def main():
    print("Mental Health Support Chatbot")
    print("=============================")
    print("I'm here to listen and support you. While I'm not a replacement for")
    print("professional mental health care, I can provide a space for you to")
    print("share your thoughts and feelings.")
    print("Type 'exit' to end the conversation.")
    print("\n")

    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            print("\nTake care! Remember that professional mental health")
            print("resources are available if you need them.")
            break

        try:
            # Get the response from the conversation chain
            response = conversation.predict(input=user_input)
            print("\nBot:", response.strip(), "\n")

        except Exception as e:
            print("\nI apologize, but I'm having trouble processing that.")
            print("Please try rephrasing or take a moment before continuing.\n")

if __name__ == "__main__":
    main()