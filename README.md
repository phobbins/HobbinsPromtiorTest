## Installation

### Prerequisites

- Python 3.8 or higher
- FastAPI
- Uvicorn
- LangChain and its dependencies
- [Ollama](https://ollama.com) with the llama2 model server running

### Setup

1. Install the required packages:

    ```bash
    pip install fastapi uvicorn langchain langchain_core langchain_community langchain_huggingface
    ```

2. Set the `TAVILY_API_KEY` environment variable:

    ```python
    import getpass
    import os

    os.environ["TAVILY_API_KEY"] = getpass.getpass()
    ```
    It's free to make an account


3. Ensure Ollama and the llama2 model server are running:

    - Follow the instructions on the [Ollama website](https://ollama.com) to set up and run the llama2 model server.


## Usage

1. Start the FastAPI server:

   Run 'run_server.py' 

2. Access the server at `http://localhost:8000/agent/playground/`
