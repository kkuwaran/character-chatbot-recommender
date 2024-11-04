# Character Chatbot Recommender

**Project name:** character-chatbot-recommender

**Description:** A character chatbot that suggests fictional characters based on personality traits or keywords. 
Utilizing OpenAI embeddings, this tool finds characters that closely match user-input traits through similarity scoring.

---

### Overview

This project builds a chatbot to recommend fictional characters from theater, television, and film productions based on personality traits, keywords, or MBTI types. 
The recommendations are powered by embeddings from character descriptions, allowing for a Retrieval-Augmented Generation (RAG) model that provides accurate matches to user queries.

---

### Key Features

- **Character Recommendations**: Finds characters aligned with specified traits or keywords.
- **Embeddings for Similarity Matching**: Utilizes OpenAI's embeddings to calculate similarity between user queries and character descriptions.
- **Compatibility**: Tested with OpenAI API versions `0.28.0` and `1.52.1`.

---

### Repository Structure

```plaintext
character-chatbot-recommender/
├── data/
│   ├── character_descriptions.csv                  # Input data with character details
│   ├── character_descriptions_embeddings_v0.csv    # Embeddings from OpenAI API v0.28.0
│   └── character_descriptions_embeddings_v1.csv    # Embeddings from OpenAI API v1.52.1
├── custom_chatbot.ipynb                            # Jupyter notebook for embedding the dataset and RAG model
├── tester.ipynb                                    # Notebook for testing individual sub-functions
├── openai_api_wrapper.py                           # OpenAI API client wrapper
├── data_processor.py                               # Data cleaning and preprocessing functions
├── embedding_generator.py                          # Functions for generating embeddings
├── rag_generator.py                                # RAG model code for generating recommendations
├── README.md                                       # Project documentation
├── requirements.txt                                # Python dependencies
└── LICENSE                                         # License for the project
```

### Data

**Input Dataset**: `data/character_descriptions.csv` - A dataset containing fictional character descriptions with fields such as `name`, `description`, `medium`, and `setting`.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kkuwaran/character-chatbot-recommender.git
   cd character-chatbot-recommender
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure OpenAI API key**: Set up your OpenAI API key by either:
   - Adding it as an environment variable:
     ```bash
     export OPENAI_API_KEY='your-api-key'
     ```
   - Replacing `os.environ.get("OPENAI_API_KEY")` with your API key string in `custom_chatbot.ipynb`.

---

## Usage

1. **Generate Embeddings (optional)**
   To embed character descriptions, run the second cell in `custom_chatbot.ipynb`.
   *Note:* If `character_descriptions_embeddings_v0.csv` (for  OpenAI API v0.28.0) or `character_descriptions_embeddings_v1.csv` (for OpenAI API v1.52.1) already exists in the data directory, you can skip this step.
2. **Run the Chatbot**
   Use the remaining cells in `custom_chatbot.ipynb` to set the query parameters, query the dataset, and get character recommendations.

---
