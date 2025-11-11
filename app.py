import streamlit as st
import os
import json
import time
import requests

# --- Configuration ---
# NOTE: The user requested 'gemini-2.0-flash'. We are using the most capable and
# recommended model for grounded search, gemini-2.5-flash-preview-09-2025.
MODEL_NAME = "gemini-2.0-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# The API Key is expected to be provided by the environment.
# When running outside of this environment, set the GEMINI_API_KEY environment variable.
API_KEY = "AIzaSyCHtoqd0rHgFQ5CKjrFfsUJ0MywuKfxpL8"

# System instruction to enforce the structured equity research report format
EQUITY_RESEARCH_SYSTEM_PROMPT = """
You are a professional Equity Research Analyst. Your task is to provide structured and objective financial and business summaries based on real-time data retrieved from Google Search.
When a user asks about a publicly traded company (e.g., "Analyze Apple," or "What is the outlook for Tesla?"), generate a detailed report using clear Markdown headings and bullet points.

The structure of your response MUST include the following sections, populated with current data:

1.  **Company Overview:** (Industry, Primary Business, Market Cap, Ticker)
2.  **Recent Financial Performance:** (Key figures like Revenue, Net Income, or EPS from the latest available quarterly or annual report)
3.  **Recent News & Catalysts:** (1-3 major recent developments that could affect the stock price)
4.  **Outlook & Analyst Sentiment:** (A brief, objective summary of the consensus future outlook)
5.  **Key Risks:** (1-2 major business or market risks)

If the query is not about a company analysis, answer it conversationally. If you cannot find current data, state that clearly.
"""

# --- Streamlit Setup ---
st.set_page_config(page_title="Equity Research Gemini Analyst", layout="wide")
st.title("💰 Gemini Equity Research Analyst")
st.caption("Powered by Gemini and Google Search Grounding for real-time analysis.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add an initial greeting message
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I am your AI Equity Research Analyst. Ask me to analyze any publicly traded company (e.g., 'Analyze Google') to get a structured report."})


def call_gemini_api(prompt, history, system_instruction):
    """
    Calls the Gemini API with search grounding enabled and handles retries.
    """
    if not API_KEY:
        st.error("API Key is missing. Please set the GEMINI_API_KEY environment variable.")
        return "Error: API Key is not configured."

    # Prepare chat history for the API call
    # The history should include all previous user/assistant turns
    contents = []
    for msg in history:
        # The first assistant message is the greeting and shouldn't be included as a turn
        if not (msg["role"] == "assistant" and msg["content"].startswith("Hello!")):
            # Convert Streamlit roles to Gemini roles
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    
    # Append the current user prompt
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    # Construct the API payload
    payload = {
        "contents": contents,
        "tools": [{"google_search": {}}],  # Enable Google Search grounding
        "systemInstruction": {"parts": [{"text": system_instruction}]}
    }

    headers = {'Content-Type': 'application/json'}
    
    # Exponential backoff logic for retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_URL}?key={API_KEY}",
                headers=headers,
                data=json.dumps(payload),
                timeout=30  # Increased timeout for complex searches
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            # Extract the generated text
            text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'Could not generate content.')

            # Extract grounding sources
            sources = []
            grounding_metadata = candidate.get('groundingMetadata')
            if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                sources = grounding_metadata['groundingAttributions']
            
            return text, sources
            
        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"API call failed with {response.status_code}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return f"An HTTP error occurred: {e}", []
        except requests.exceptions.RequestException as e:
            return f"A connection error occurred: {e}", []
        except Exception as e:
            return f"An unexpected error occurred during API processing: {e}", []
    
    return "Failed to get a response from the API after multiple retries.", []


# --- Chat Interface Logic ---

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter a company name or a research question..."):
    # 1. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data and generating report..."):
            # Call the API function
            report_text, sources = call_gemini_api(
                prompt, 
                st.session_state.messages,
                EQUITY_RESEARCH_SYSTEM_PROMPT
            )
            
            # Display the main generated text
            st.markdown(report_text)
            
            # Display citations if available
            if sources:
                st.markdown("---")
                st.markdown("**Sources Used:**")
                for i, source in enumerate(sources):
                    if source.get('web'):
                        uri = source['web'].get('uri', '#')
                        title = source['web'].get('title', 'Link')
                        st.markdown(f"{i+1}. [{title}]({uri})")
            
            # 3. Add assistant response to chat history
            # Append the full response (text + sources) to history
            full_response_content = report_text
            if sources:
                 # Reconstruct the source display into a string for history preservation
                source_markdown = "\n\n---\n**Sources Used:**\n" + \
                                  "\n".join([f"{i+1}. [{s['web'].get('title', 'Link')}]({s['web'].get('uri', '#')})" 
                                             for i, s in enumerate(sources) if s.get('web')])
                full_response_content += source_markdown
                
            st.session_state.messages.append({"role": "assistant", "content": full_response_content})

# Sidebar cleanup button
st.sidebar.title("History")
if st.sidebar.button("Clear Chat History", type="secondary"):
    st.session_state.messages = []
    # Re-add the initial greeting
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I am your AI Equity Research Analyst. Ask me to analyze any publicly traded company (e.g., 'Analyze Google') to get a structured report."})
    st.rerun()

st.sidebar.info("Remember to ask specific questions about companies for best results, e.g., 'Give me an analysis of Microsoft Corp.'")