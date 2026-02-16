import pickle
from pathlib import Path
from typing import Any, Dict, List


from gitsource import GithubRepositoryDataReader

from minsearch import Index, Highlighter, Tokenizer
from minsearch.tokenizer import DEFAULT_ENGLISH_STOP_WORDS


class SearchTools:
    """
    Provides search and file retrieval utilities over an indexed data store.
    """

    def __init__(
        self,
        index: Index,
        highlighter: Highlighter,
        file_index: Dict[str, str]
    ) -> None:
        """
        Initialize the SearchTools instance.

        Args:
            index: Searchable index providing a `search` method.
            highlighter: Highlighter used to annotate search results.
            file_index (Dict[str, str]): Mapping of filenames to file contents.
        """
        self.index = index
        self.highlighter = highlighter
        self.file_index = file_index

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the index for results matching a query and highlight them.

        Args:
            query (str): The search query to look up in the index.

        Returns:
            List[Dict[str, Any]]: A list of highlighted search result objects.
        """
        search_results = self.index.search(
            query=query,
            num_results=5
        )

        return self.highlighter.highlight(query, search_results)

    def get_file(self, filename: str) -> str:
        """
        Retrieve a file's contents by filename.

        Args:
            filename (str): The filename of the file to retrieve.

        Returns:
            str: The file contents if found, otherwise an error message.
        """
        if filename in self.file_index:
            return self.file_index[filename]
        return f"file {filename} does not exist"



def download_documentation_data():
    reader = GithubRepositoryDataReader(
        repo_owner="evidentlyai",
        repo_name="docs",
        allowed_extensions={"md", "mdx"},
    )
    files = reader.read()

    parsed_docs = [doc.parse() for doc in files]

    return parsed_docs


def build_index(parsed_docs):
    index = Index(
        text_fields=["title", "description", "content"],
        keyword_fields=["filename"]
    )
    index.fit(parsed_docs)
    return index


def create_highlighter():
    stopwords = DEFAULT_ENGLISH_STOP_WORDS | {'evidently'}

    highlighter = Highlighter(
        highlight_fields=['content'],
        max_matches=3,
        snippet_size=50,
        tokenizer=Tokenizer(stemmer='snowball', stop_words=stopwords)
    )

    return highlighter


def create_documentation_tools():
    parsed_docs = download_documentation_data()
    index = build_index(parsed_docs)
    highlighter = create_highlighter()

    file_index = {doc['filename']: doc['content'] for doc in parsed_docs}

    return SearchTools(
        index=index,
        highlighter=highlighter,
        file_index=file_index
    )


def create_documentation_tools_cached():
    cache_dir = Path('.cache')
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / 'documentation_tools.pkl'

    if cache_file.exists():
        print(f"Loading documentation tools from {cache_file}...")

        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        return SearchTools(
            index=cache['index'],
            highlighter=create_highlighter(),
            file_index=cache['file_index']
        )

    print("Cache not found, creating documentation tools...")

    parsed_docs = download_documentation_data()

    index = build_index(parsed_docs)
    file_index = {doc['filename']: doc['content'] for doc in parsed_docs}

    cache = {
        'index': index,
        'file_index': file_index
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)

    print(f"Documentation tools cached to {cache_file}.")

    return SearchTools(
        index=index,
        highlighter=create_highlighter(),
        file_index=file_index
    )
