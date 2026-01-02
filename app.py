
import gradio as gr
from src.rag_pipeline import ComplaintRAG

# Initialize RAG System (Load once)
print("Loading RAG System for UI...")
rag = ComplaintRAG()

def chat_function(message, history):
    # 1. Retrieve sources for display
    docs = rag.retrieve_only(message)
    
    # 2. Generate Answer
    # Note: query() internally calls retrieval again. 
    # For a production system we'd optimize to pass docs, but this is fine for now.
    answer = rag.query(message)
    
    # 3. Format Sources
    sources_html = "<br><hr><h4>Sources:</h4>"
    for i, doc in enumerate(docs):
        # Extract metadata safely
        product = doc.metadata.get('Product', 'Unknown Product')
        company = doc.metadata.get('Company', 'Unknown Company')
        c_id = doc.metadata.get('Complaint ID', 'N/A')
        state = doc.metadata.get('State', 'N/A')
        date = doc.metadata.get('Date received', 'N/A')
        
        # Snippet (first 200 chars)
        snippet = doc.page_content[:200] + "..."
        
        sources_html += f"""
        <div style="margin-bottom: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
            <b>{i+1}. {product}</b> <span style="font-size: 0.8em; color: gray;">({company}, {state})</span><br>
            <span style="font-size: 0.9em; font-style: italic;">"{snippet}"</span><br>
            <span style="font-size: 0.8em;">ID: {c_id} | Date: {date}</span>
        </div>
        """
    
    # Combine Answer and Sources
    final_output = f"{answer}\n{sources_html}"
    return final_output

# Create Gradio Interface
with gr.Blocks(title="CrediTrust Intelligent Complaint Analysis") as demo:
    gr.Markdown("# CrediTrust Financial - Complaint Insights AI")
    gr.Markdown("Ask questions about customer complaints related to Credit Cards, Loans, Savings, and Money Transfers.")
    
    chatbot = gr.Chatbot(label="Conversation")
    msg = gr.Textbox(label="Your Question", placeholder="e.g., Why are people complaining about credit card late fees?")
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = chat_function(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(share=False)
