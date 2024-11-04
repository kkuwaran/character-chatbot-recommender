from typing import List
import time
import openai
import tiktoken



class OpenAIClient:
    def __init__(self, api_base: str, api_key: str):
        """Initialize the OpenAI client based on the installed OpenAI version."""

        self.api_base = api_base
        self.api_key = api_key

        # Initialize the OpenAI client and models (defined in setup_client)
        self.version_index = None
        self._client = None
        self._embeeding_model = None
        self._completion_model = None

        # Get the OpenAI version and set up the client
        self._openai_version = self.get_openai_version()
        self.tokenizer = self.setup_client()


    @staticmethod
    def get_openai_version() -> str:
        """Get the version of the OpenAI library that is currently installed."""

        try:
            version = openai.__version__
        except AttributeError:
            import pkg_resources
            version = pkg_resources.get_distribution("openai").version
        return version


    def setup_client(self) -> tiktoken.core.Encoding:
        """
        Set up the OpenAI API version-dependent variables and models.
        
        Returns:
        tokenizer: The tokenizer for the OpenAI API.
        """

        # Print the OpenAI version
        print(f"OpenAI version: {self._openai_version}")

        # Set up the OpenAI API based on the version
        if self._openai_version.startswith("0"):
            self.version_index = 0
            self._embeeding_model = "text-embedding-ada-002"
            self._completion_model = "gpt-3.5-turbo-instruct"
            tokenizer = tiktoken.get_encoding("cl100k_base")

            openai.api_base = self.api_base
            openai.api_key = self.api_key

        elif self._openai_version.startswith("1"):
            self.version_index = 1
            self._embeeding_model = "text-embedding-3-small"
            self._completion_model = "gpt-4o"
            tokenizer = tiktoken.encoding_for_model(self._completion_model)

            self.client = openai.OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )
        else:
            raise ValueError("Unsupported OpenAI version")

        return tokenizer


    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using the OpenAI API."""

        model = self._embeeding_model
        if self.version_index == 0:
            response = openai.Embedding.create(input=texts, engine=model)
            embeddings = [item.embedding for item in response["data"]]
        else:
            response = self.client.embeddings.create(input=texts, model=model)
            embeddings = [item.embedding for item in response.data]
        return embeddings


    def query_openai(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        """Query the OpenAI API using a constructed prompt."""

        model = self._completion_model
        if self.version_index == 0:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            time.sleep(5)
            text_reponse = response.choices[0].text.strip().strip("\n")
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            time.sleep(5)
            text_reponse = response.choices[0].message.content.strip().strip("\n")

        return text_reponse