import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize the FastAPI Web Server
app = FastAPI(title="Gemini Summarization Agent")

# 2. Define the input format we expect from the user
class SummarizeRequest(BaseModel):
    text: str

# 3. Create the HTTP Endpoint
@app.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    try:
        # Get the API key from environment variables (we will set this securely in Render later)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set.")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Use the fast, free tier model
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        # Instruct the agent on its specific task
        prompt = f"You are a highly capable text summarization agent. Summarize the following text clearly and concisely in one short paragraph:\n\n{request.text}"
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        
        # Return the valid JSON response
        return {"summary": response.text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. A simple health check endpoint for the main URL
@app.get("/")
async def root():
    return {"status": "Agent is running!", "instructions": "Send a POST request to /summarize with JSON payload {'text': 'your text here'}"}