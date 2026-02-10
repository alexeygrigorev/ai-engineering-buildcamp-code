from typing import Any, Dict, List


class SearchIndexTools:
    def __init__(self, index: Any) -> None:
        self.index = index

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the documentation database for relevant results based on a query string.

        Args:
            query (str): The search query to look up in the index.

        Returns:
            List[Dict[str, Any]]: A list of search result objects returned by the index.
        """
        results = self.index.search(
            query=query,
            num_results=5
        )
        return results

    def add_entry(
        self,
        filename: str,
        title: str,
        description: str,
        content: str
    ) -> str:
        """
        Add a new documentation entry to the index.

        Args:
            filename (str): The source filename associated with the entry.
            title (str): The title of the documentation entry.
            description (str): A short description summarizing the entry.
            content (str): The full content of the documentation entry.

        Returns:
            str: A status message indicating the result of the operation ("OK").
        """
        entry: Dict[str, Any] = {
            'start': 0,
            'content': content,
            'title': title,
            'description': description,
            'filename': filename,
        }
        self.index.append(entry)
        return "OK"
