import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import re
from groq import Groq

class BareMetalRAG:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        # 1. Initialize the Bi-Encoder Model
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Get the mathematical dimension of the output vector (e.g., 384 for bge-small)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # 2. Initialize the FAISS Vector Database
        # We use IndexFlatIP (Inner Product). If vectors are L2-normalized, 
        # Inner Product is mathematically identical to Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # We need a separate dictionary to map FAISS integer IDs back to our text chunks
        self.chunk_map = {}
        self.current_id = 0

        # 3. Initialize the Cross-Encoder (The highly accurate magnifying glass)
        reranker_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"Loading Re-Ranker model: {reranker_name}...")
        self.reranker = CrossEncoder(reranker_name)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts raw text from a PDF, cleaning up physical layout artifacts."""
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            text = page.get_text("text")
            # Clean up whitespace and hyphenation that breaks semantic meaning
            text = re.sub(r'\s+', ' ', text).strip()
            full_text += text + " "
        return full_text

    def sliding_window_chunker(self, text: str, chunk_size: int = 250, overlap: int = 50) -> list[str]:
        """
        Splits text into overlapping chunks. 
        In production, use a tokenizer (like tiktoken), but word-splitting 
        demonstrates the raw sliding window mechanics perfectly.
        """
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]

        # The math of the sliding window stride
        stride = chunk_size - overlap
        
        for i in range(0, len(words), stride):
            # Slice the window
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
            
            # Break if we've hit the end of the document
            if i + chunk_size >= len(words):
                break
                
        return chunks

    def ingest_document(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 200):
        """The complete ingestion pipeline: Extract -> Chunk -> Embed -> Index."""
        print(f"Ingesting {pdf_path}...")
        
        # 1. Extract and Chunk
        raw_text = self.extract_text_from_pdf(pdf_path)
        chunks = self.sliding_window_chunker(raw_text, chunk_size, overlap)
        print(f"Document split into {len(chunks)} chunks.")
        
        # 2. Generate Embeddings (The Forward Pass)
        # We MUST normalize the embeddings to map Inner Product to Cosine Similarity
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        
        # FAISS requires the numpy array to be strictly float32
        embeddings_matrix = np.array(embeddings).astype('float32')
        
        # 3. Store in Vector Database
        self.index.add(embeddings_matrix)
        
        # 4. Map the FAISS IDs to the text chunks
        for i, chunk in enumerate(chunks):
            self.chunk_map[self.current_id + i] = chunk
            
        self.current_id += len(chunks)
        print(f"Ingestion complete. Total vectors in DB: {self.index.ntotal}")

    def query_database(self, query: str, initial_k: int = 15, final_k: int = 3) -> list[dict]:
        """Phase 1: Fast Retrieval, Phase 2: Deep Verification."""
        # ==========================================
        # PHASE 1: Fast Retrieval (Bi-Encoder)
        # ==========================================
        query_vector = self.model.encode([query], normalize_embeddings=True)
        query_matrix = np.array(query_vector).astype('float32')
        scores, indices = self.index.search(query_matrix, initial_k)
        
        # Gather the top 15 candidate chunks from the FAISS index
        candidates = []
        for idx in indices[0]:
            if idx != -1: 
                candidates.append(self.chunk_map[idx])
                
        if not candidates:
            return []

        # ==========================================
        # PHASE 2: Deep Verification (Cross-Encoder)
        # ==========================================
        # Create pairs to feed into the cross-encoder: [[query, chunk1], [query, chunk2]...]
        cross_inp = [[query, chunk] for chunk in candidates]
        
        # The model scores how well each chunk actually answers the query
        cross_scores = self.reranker.predict(cross_inp)
        
        # Combine the chunks with their new accuracy scores
        scored_candidates = list(zip(candidates, cross_scores))
        
        # Sort them from highest score to lowest score
        sorted_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
        
        # ==========================================
        # PHASE 3: Return the Top K
        # ==========================================
        results = []
        for chunk, score in sorted_candidates[:final_k]:
            results.append({
                "score": float(score),
                "text": chunk
            })
            
        return results

# ==========================================
# Execution Example
# ==========================================# ==========================================
# Execution Example
# ==========================================
# ==========================================
# Execution Example (The Agentic Workflow)
# ==========================================
if __name__ == "__main__":
    import yfinance as yf
    import json
    import os
    from dotenv import load_dotenv
    import concurrent.futures

    load_dotenv()
    
    # Initialize the Groq Cloud Client
    # Replace with your new, secure API key!
    llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Initialize our bare-metal RAG
    rag = BareMetalRAG()
    
    rag.ingest_document("data/AAPL_2025_10K_Report.pdf")

    print("\n✅ Multi-Tool Agent ready! Type 'exit' to quit.")
    
    # 1. Initialize memory OUTSIDE the loop
    chat_history = []

    while True:
        user_query = input("\nAsk a question (Stock, Math, or Book): ") 
        
        if user_query.lower() == 'exit':
            print("Shutting down Agent...")
            break
            
        chat_history.append({"role": "user", "content": user_query})

        # ==========================================
        # STEP 1: THE ROUTER (Brain)
        # ==========================================
        print("\n[Agent Thinking... deciding which tool to use]")
        
        # Convert the last few chat turns into a readable string for the router
        recent_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])

        routing_prompt = f"""You are an intelligent routing agent. Analyze the user's query and decide which tool to use.
        You MUST respond ONLY with a valid JSON object.
        
        Available Tools:
        1. "search_vector_db": For questions about the uploaded book. Required parameter: "query" (This must be a simple text string of what to search for).
        2. "get_stock_price": For live stock prices. Required parameter: "ticker" (MUST append .NS for Indian stocks).
        3. "calculate_math": For math calculations. Required parameter: "expression" (valid Python math).
        4. "none": Use this if the user's query is off-topic (like weather), a general greeting, or does not require any external tools. Required parameter: "none".    
        <recent_conversation_context>
        {recent_history}
        </recent_conversation_context>

        <user_latest_query>
        {user_query}
        </user_latest_query>
        
        Example Output:
        {{"tool": "get_stock_price", "parameter": "RELIANCE.NS"}}
        """
        
        # Force LLaMA 3 to return JSON using Groq's response_format
        routing_response = llm_client.chat.completions.create(
            messages=[{"role": "system", "content": routing_prompt}],
            model="llama-3.1-8b-instant",
            temperature=0, # Keep temp at 0 for strict, deterministic logic
            response_format={"type": "json_object"} 
        )
        
        routing_decision = routing_response.choices[0].message.content
        
        # ==========================================
        # STEP 2: TOOL EXECUTION (Hands)
        # ==========================================
        try:
            decision_data = json.loads(routing_decision)
            tool = decision_data.get("tool")
            param = decision_data.get("parameter")
            
            print(f"[Action: Executing '{tool}' with parameter: '{param}']")
            
            tool_result = ""
            
            if tool == "search_vector_db":
                results = rag.query_database(param, initial_k=15, final_k=3)
                tool_result = "\n\n".join([r['text'] for r in results])
                if not tool_result.strip():
                    tool_result = "I searched the book but could not find answer for the question. The question asked is not related to the book."
                    
            elif tool == "get_stock_price":
                # Define the network call as a helper function
                def fetch_price():
                    stock = yf.Ticker(param)
                    return stock.fast_info['last_price']
                
                # Execute the network call inside a threaded stopwatch
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(fetch_price)
                    try:
                        # Wait maximum 5 seconds for the API to respond
                        price = future.result(timeout=5.0) 
                        tool_result = f"The live price of {param} is ₹{price:.2f}"
                        
                    except concurrent.futures.TimeoutError:
                        # This triggers if the 5 seconds run out
                        tool_result = "System Error: The Stock API timed out. Please apologize to the user and suggest trying again later."
                        
                    except Exception as e:
                        # This triggers if the ticker is invalid or another error occurs
                        tool_result = f"Could not fetch stock data for {param}. Make sure it is a valid ticker."
            elif tool == "calculate_math":
                try:
                    result = eval(param) # Using eval for bare-metal testing
                    tool_result = f"The result of the calculation is {result}"
                except Exception as e:
                    tool_result = f"Math error: Could not calculate {param}."
                    
            elif tool == "none":
                # The agent realized no tools are needed! 
                tool_result = "No tools were used. Answer the user's question directly based on your own general knowledge, or politely explain what your capabilities are."
            
            else:
                tool_result = "Error: Model hallucinated an unknown tool."
                
        except json.JSONDecodeError:
            tool_result = "System Error: LLaMA 3 failed to return valid JSON."
        
        # ==========================================
        # STEP 3: FINAL SYNTHESIS (Voice)
        # ==========================================
        print("[Synthesizing final conversational response...]")
        
        # 1. Define the system instructions
        synthesis_system_prompt = {
            "role": "system", 
            "content": f"""You are a helpful, conversational AI assistant.
            You MUST speak in natural, human language. Do NOT output JSON format.
            
            You just used your internal tools to fetch this raw data: 
            <tool_data>
            {tool_result}
            </tool_data>
            
            If the <tool_data> contains an error message, politely apologize to the user and explain that the system could not fetch the data. 
            Otherwise, answer the user's latest question naturally using ONLY the <tool_data> and the conversation history.
            """
        }
        
        # 2. Combine the system instructions with our rolling memory
        synthesis_messages = [synthesis_system_prompt] + chat_history
        
        # 3. Pass the combined list DIRECTLY to the API
        final_completion = llm_client.chat.completions.create(
            messages=synthesis_messages, # <-- No extra brackets here!
            model="llama-3.1-8b-instant",
            temperature=0.4
        )
        
        final_answer = final_completion.choices[0].message.content
        
        print("\n🤖 AI Answer:")
        print(final_answer)
        print("-" * 50)
        
        # # 4. Save the AI's final answer to the memory bank for the NEXT loop
        # chat_history.append({"role": "assistant", "content": final_answer})