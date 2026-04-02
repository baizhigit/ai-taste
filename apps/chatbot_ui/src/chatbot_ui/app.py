import streamlit as st
import requests
from chatbot_ui.core.config import config


def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}

    
# ## Lets create a sidebar with a dropdown for the model list and providers
# with st.sidebar:
#     st.title("Settings")

#     #Dropdown for model
#     provider = st.selectbox("Provider", ["OpenAI", "Groq", "Google"])
#     if provider == "OpenAI":
#         model_name = st.selectbox("Model", ["gpt-5-nano", "gpt-5-mini"])
#     elif provider == "Groq":
#         model_name = st.selectbox("Model", ["llama-3.3-70b-versatile"])
#     else:
#         model_name = st.selectbox("Model", ["gemini-2.5-flash"])

#     # Save provider and model to session state
#     st.session_state.provider = provider
#     st.session_state.model_name = model_name


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        output = api_call("post", f"{config.API_URL}/rag", json={"query": prompt})
        response_data = output[1]
        answer = response_data["answer"]
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})