"""
Search tools for agentic RAG documentation assistant.

Provides search and file retrieval over an indexed documentation store.
"""

from typing import Any, Dict, List

from minsearch import AppendableIndex, Highlighter, Tokenizer
from minsearch.tokenizer import DEFAULT_ENGLISH_STOP_WORDS
from gitsource import GithubRepositoryDataReader


class SearchTools:
    """
    Provides search and file retrieval utilities over an indexed data store.
    """

    def __init__(
        self,
        index: AppendableIndex,
        highlighter: Highlighter,
        file_index: Dict[str, str],
    ) -> None:
        """
        Initialize SearchTools instance.
    
        Args:
            index: Searchable index providing a `search` method.
            highlighter: Highlighter instance for highlighting search results.
            file_index: Mapping of filenames to file contents.
        """
        self.index = index
        self.highlighter = highlighter
        self.file_index = file_index


    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search index for results matching a query and optionally highlight them.

        Args:
            query: The search query to look up in index.

        Returns:
            A list of search result objects.
        """
        search_results = self.index.search(query=query, num_results=5)
        return self.highlighter.highlight(query, search_results)

    def get_file(self, filename: str) -> str:
        """
        Retrieve a file's contents by filename.

        Args:
            filename: The filename of file to retrieve.

        Returns:
            The file contents if found, otherwise an error message.
        """
        if filename in self.file_index:
            return self.file_index[filename]
        return f"file {filename} does not exist"


def create_search_tools() -> SearchTools:
    """
    Factory function to create SearchTools pre-configured with a GitHub repo.

    Args:
        repo_owner: GitHub repository owner.
        repo_name: GitHub repository name.
        allowed_extensions: Set of file extensions to include (e.g. {"md", "mdx"}).
        enable_highlighting: Whether to enable search result highlighting.

    Returns:
        Configured SearchTools instance.
    """

    reader = GithubRepositoryDataReader(
        repo_owner="evidentlyai",
        repo_name="docs",
        allowed_extensions={"md", "mdx"},
    )
    files = reader.read()
    parsed_docs = [doc.parse() for doc in files]

    index = AppendableIndex(
        text_fields=["title", "description", "content"],
        keyword_fields=["filename"]
    )
    index.fit(parsed_docs)

    file_index = {doc["filename"]: doc["content"] for doc in parsed_docs}

    stopwords = DEFAULT_ENGLISH_STOP_WORDS | {"evidently"}
    highlighter = Highlighter(
        highlight_fields=["content"],
        max_matches=3,
        snippet_size=50,
        tokenizer=Tokenizer(stemmer="snowball", stop_words=stopwords)
    )

    return SearchTools(index, highlighter, file_index)
