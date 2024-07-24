## Installation

### Prerequisites

- Python 3.8 or higher
- FastAPI
- Uvicorn
- LangChain and its dependencies

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

## Usage

1. Start the FastAPI server:

   Run run_server.py 

2. Access the server at `http://localhost:8000/agent/playground/`
