import gradio as gr
from chatbot import load_qa_chain

qa_chain = load_qa_chain()

def ask_question(query):
    response = qa_chain(query)
    answer = response["result"]
    return answer

iface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(lines=3, placeholder="Ask a question based on the document..."),
    outputs="text",
    title="RAG Chatbot",
    description="Ask questions based on uploaded content using a Retrieval-Augmented Generation pipeline."
)

if __name__ == "__main__":
    iface.launch()
