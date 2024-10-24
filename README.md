# OpenFDA Chatbot

## Overview

The **OpenFDA Recall Chatbot** is an interactive chatbot application designed to assist users in retrieving and exploring product recall data from the FDA. It provides users with detailed information about product recalls, such as product codes, batch numbers, reasons for recall, and more, by using structured JSON data obtained from the FDA recall database.

This project aims to simplify access to FDA recall information through natural language queries, making it easier for users to find specific recall data.

## Features

- **Natural Language Queries**: Users can ask questions in plain English to retrieve recall information.
- **Structured Recall Data**: Fetches recall data with fields like product code, batch numbers, recall reasons, and dates.
- **Data Inconsistency Handling**: The chatbot is designed to manage inconsistencies in the data across different product recall fields.
- **User-Friendly Interface**: Clean and easy-to-use interface for both developers and end users.


### Key Files

- **`app.py`**: Contains the main application logic for running the chatbot. Integrates with Streamlit for the user interface.
- **`langCode.py`**: Contains the functions that handle natural language processing and interaction with the language model.
- **`RAG.py`**: Implements retrieval-augmented generation (RAG) to pull relevant recall information based on user queries.
- **`requirements.txt`**: Contains the list of Python dependencies required to run the chatbot.

## Installation

To set up the OpenFDA Chatbot locally, follow these steps:

### Prerequisites

- Python 3.12 or higher
- `pip` (Python package manager)

### Steps

1. **Clone the repository**:

2. **Set up a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
5. **Change Directory**:
    ```bash
    langCode.py -> line 6
    ```

4. **Run the application**:
    ```bash
    streamlit run app.py
    ```

The application will be available at `http://localhost:8501` in your web browser.

## Usage

1. **Ask Questions**: Users can ask questions like:
    - "Pediatric Two-Lumen Central Venous Catheterization why was this recalled, and give the Batch Numbers for the same."
    - "How many products were recalled in 2024?"
    - "Which products were recalled in 2023 due to mislabeling?"

2. **Get Recall Details**: The chatbot will retrieve and display relevant recall information in response to your queries, showing product codes, recall reasons, and other details.

## Technologies Used

- **Python**: Main programming language for building the chatbot.
- **Streamlit**: Used for creating the user interface.
- **LangChain**: To handle natural language processing and response generation.
- **FDA Recall Data**: JSON-based structured data from the FDA recall vectorstore, pulled from openFDA API.
