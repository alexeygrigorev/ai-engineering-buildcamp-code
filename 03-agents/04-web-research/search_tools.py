
import time
from typing import List, Dict
import requests


class SearchTools:

    def __init__(self, brave_api_key: str, sleep_time: float = 1):
        self.brave_api_key = brave_api_key
        self.sleep_time = sleep_time

    def brave_search(self, query: str, num_results: int = 20) -> List[Dict[str, str]]:
        """
        Search the web
    
        Args:
            query: The search query string.
    
        Raises:
            requests.HTTPError: If the API request fails (non-200 response).
            KeyError: If the response format is unexpected.
            requests.RequestException: For network-related errors.
        """
    
        time.sleep(self.sleep_time)

        url = "https://api.search.brave.com/res/v1/web/search"
    
        headers = {
            "X-Subscription-Token": self.brave_api_key,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }
        
        params = {
            "q": query,
            "count": num_results,
        }
    
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        brave_response = response.json()
    
        search_results = brave_response['web']['results']
    
        output: List[Dict[str, str]] = []
        for r in search_results:
            output.append({
                'title': r['title'],
                'url': r['url'],
                'description': r['description'],
            })
    
        return output

    def fetch_content(self, url: str) -> str:
        """
        Retrieve webpage content.
    
        Args:
            url: The target webpage URL.
    
        Returns:
            The textual content of the page as a UTF-8 decoded string.
    
        Raises:
            requests.HTTPError: If the HTTP request returns a non-success status code.
            requests.RequestException: For network-related errors.
            UnicodeDecodeError: If the response body cannot be decoded as UTF-8.
        """
        jina_prefix = "https://r.jina.ai"
        final_url = f"{jina_prefix}/{url}"
    
        response = requests.get(final_url)
        response.raise_for_status()
    
        return response.content.decode("utf-8")
