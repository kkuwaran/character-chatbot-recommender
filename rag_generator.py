import os
import ast
import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.spatial import distance
import tiktoken



class RAGModel:
    """A class for performing Retrieval-Augmented Generation (RAG) to answer queries based on context embeddings."""

    DEFAULT_INSTRUCTION = """Please answer the question using the context provided below."""
    DEFAULT_LLM_KWARGS = {
        'max_input_tokens': 500,
        'max_output_tokens': 150,
        'temperature': 0.7,
        'top_p': 0.9,
    }
    

    def __init__(self, client, file_names: List[str], embedding_col: str = "embedding"):
        """Initialize the RAGModel with the specified client and dataframes."""

        # Initialize client attributes
        assert hasattr(client, "get_embeddings"), "Client must have 'get_embeddings' method"
        assert hasattr(client, "query_openai"), "Client must have 'query_openai' method"
        assert hasattr(client, "tokenizer"), "Client must have 'tokenizer' attribute"
        self.embedding_fn = client.get_embeddings
        self.query_openai_fn = client.query_openai
        self.tokenizer = client.tokenizer

        # Initialize dataframe processing attributes
        self.file_names = file_names
        self.embedding_col = embedding_col

        # Miscellaneous attributes
        self.verbose = None
        
        # Get dataframe from the provided files
        self.df = self.load_and_prepare_dataframes()


    def load_and_prepare_dataframes(self) -> pd.DataFrame:
        """Load, validate, and prepare dataframes by converting embeddings to arrays."""

        # Extract embedding column name
        emb_col_name = self.embedding_col
        
        # Load and validate dataframes
        dfs = []
        columns = None
        for file_name in self.file_names:
            # Ensure file exists and is a CSV
            assert os.path.exists(file_name), f"File {file_name} not found"
            assert file_name.endswith(".csv"), "Only CSV files are supported"
            
            # Load data
            df = pd.read_csv(file_name)
            
            # Validate consistency of columns across files
            if columns is None:
                columns = df.columns
            else:
                assert (columns == df.columns).all(), "All dataframes must have the same columns"
            
            # Ensure embedding column exists and convert string embeddings to arrays
            assert emb_col_name in df.columns, f"{emb_col_name} column not found in {file_name}"
            df[emb_col_name] = df[emb_col_name].apply(ast.literal_eval).apply(np.array)
            
            dfs.append(df)
        
        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # Display combined dataframe shape
        print(f"Combined dataframe has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns")
        return combined_df


    def compute_cosine_distances(self, query: str) -> pd.DataFrame:
        """Compute cosine distances between the query embedding and each row in the dataframe."""

        # Create a copy of the dataframe
        df = self.df.copy()
        
        # Get the query embedding
        query_embedding = self.embedding_fn([query])[0]
        
        # Calculate cosine distances for each row
        consine_distance_fn = lambda x: distance.cosine(x, query_embedding)
        df["cosine_distance"] = df[self.embedding_col].apply(consine_distance_fn)
        
        # Sort by cosine distance in ascending order
        df = df.sort_values("cosine_distance", ascending=True, ignore_index=True)
        
        if self.verbose:
            print(f"Top {min(10, len(df))} closest matches to the query '{query}':")
            print(df.head(10), end="\n\n")
        
        return df


    @staticmethod
    def _construct_long_context(contexts: List[str]) -> str:
        """Construct a long context string from a list of contexts for the prompt."""

        modified_contexts = [f" – {context}" for context in contexts]
        long_context = "\n".join(modified_contexts)
        return long_context


    def create_prompt(self, context_text: str, query: str) -> str:
        """Construct the prompt to query the model."""

        prompt = f"""Instruction: \n{self.instruction}\n
Context: \n{context_text}\n
Question: {query}\n
Answer:"""
        return prompt


    def construct_final_prompt(self, df: pd.DataFrame, query: str, max_input_tokens: int = 500) -> str:
        """Build a prompt with combined contexts under a token limit for the LLM."""
        
        contexts = []
        
        for _, row in df.iterrows():
            if "text" not in row:
                raise KeyError("Expected 'text' column in the dataframe")
            
            contexts.append(row["text"])
            long_context = self._construct_long_context(contexts)
            prompt = self.create_prompt(long_context, query)

            # Check token count
            token_count = len(self.tokenizer.encode(prompt))
            if self.verbose:
                print(f"Total contexts = {len(contexts)}: Total tokens = {token_count}")
            
            if token_count > max_input_tokens:
                contexts.pop()  # Remove last added context
                break

        # Create final prompt with the valid contexts
        final_long_context = self._construct_long_context(contexts)
        final_prompt = self.create_prompt(final_long_context, query)

        if self.verbose:
            print("\n=========================================")
            print(f"Final prompt with {len(contexts)} contexts: \n{final_prompt}")
            print("=========================================\n")

        return final_prompt
    

    @staticmethod
    def _format_answer(query: str, prompt: str, answer: str, format: int = 0) -> str:
        """
        Format the answer based on the specified format type.
        format = 0: Return the answer only
        format = 1: Return the question and answer
        format = 2: Return the prompt and answer
        """

        if format == 0:
            formatted_answer = answer
        elif format == 1:
            formatted_answer = f"Question: {query} \n\nAnswer: {answer}"
        elif format == 2:
            formatted_answer = f"""
==================== Prompt ====================
\n{prompt}\n
==================== Answer ====================
\n{answer}\n"""
        else:
            raise ValueError("Invalid format type. Choose from 0, 1, or 2.")
        return formatted_answer


    def get_answer(self, query: str, instruction: str = None, rag_flag: bool = True,
                   llm_kwargs: dict = None, format: int = 0, verbose: bool = False) -> str:
        """Generate an answer to a query by constructing a prompt from the context in files."""

        # Set default values for instruction and LLM kwargs if not provided
        self.instruction = instruction or self.DEFAULT_INSTRUCTION
        self.llm_kwargs = llm_kwargs or self.DEFAULT_LLM_KWARGS
        
        # Extract parameters for LLM 
        max_input_tokens = self.llm_kwargs.get('max_input_tokens', self.DEFAULT_LLM_KWARGS['max_input_tokens'])
        max_output_tokens = self.llm_kwargs.get('max_output_tokens', self.DEFAULT_LLM_KWARGS['max_output_tokens'])
        temperature = self.llm_kwargs.get('temperature', self.DEFAULT_LLM_KWARGS['temperature'])
        top_p = self.llm_kwargs.get('top_p', self.DEFAULT_LLM_KWARGS['top_p'])

        # Set verbose mode
        self.verbose = verbose

        if rag_flag:
            # Compute cosine distances between the query and each context
            df = self.compute_cosine_distances(query)
            # Construct the prompt 
            final_prompt = self.construct_final_prompt(df, query, max_input_tokens)
        else:
            final_prompt = query

        # Query the LLM model to get the answer
        answer = self.query_openai_fn(final_prompt, max_output_tokens, temperature, top_p)

        # Format the answer based on the specified format type
        formatted_answer = self._format_answer(query, final_prompt, answer, format)
        return formatted_answer