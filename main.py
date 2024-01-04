from dotenv import load_dotenv
import os
import pinecone
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
import streamlit as st
from llama_index.chat_engine.types import ChatMode


load_dotenv()

@st.cache_resource(show_spinner=False)
def getIndex() ->VectorStoreIndex:
    pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"],
    )

    indexName = 'llama'
    pineconeIndex = pinecone.Index(index_name=indexName)
    pineconeVectorStore = PineconeVectorStore(pinecone_index=pineconeIndex)
    

    # Enabling callback to identifiy times for each process
    debug = LlamaDebugHandler(print_trace_on_end=True)
    callbackManager = CallbackManager(handlers=[debug])
    serviceContext = ServiceContext.from_defaults(callback_manager=callbackManager)

    return VectorStoreIndex.from_vector_store(vector_store=pineconeVectorStore, service_context=serviceContext)

index = getIndex()

if "chatEngine" not in st.session_state.keys():
    st.session_state.chatEngine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, verbose= True)


# UI stuff
st.set_page_config(page_title="Chat with Llama", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("Chat with Llama :)")

if "messages" not in st.session_state.keys():
    st.session_state.messages=[
        {
            "role": "chatbot",
            "content": "Ask me a question :)"
        }
    ]

    
if prompt := st.chat_input("Ask here"):
    st.session_state.messages.append({"role":"user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("chatbot"):
        with st.spinner("Thinking..."):
            response = st.session_state.chatEngine.chat(message=prompt)
            st.write(response.response)
            message = {
                "role":"chatbot",
                "content":response.response
            }
            st.session_state.messages.append(message)