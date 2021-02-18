# pylint: disable=no-member
from pprint import pprint
import requests
from googleapiclient.discovery import build


api_key = "AIzaSyC2mvDtj-3VtYd9f3jzb_igeISrjO0XT50"
engine_id = "2cdde1f34b7e7c1d9"

# res keys: 'kind', 'url', 'queries', 'context', 'searchInformation', 'items'
# res['items'] item keys:
#'kind', 'title', 'htmlTitle', 'link', 'displayLink',
#'snippet', 'htmlSnippet', 'cacheId',
#'formattedUrl', 'htmlFormattedUrl', 'pagemap'


class CSE:
    def __init__(self):
        self.service = build("customsearch", "v1", developerKey=api_key)

    def search(self, query):
        """
        Search Google with the given query.

        returns: 5 results
        result (object):
            title: name of website
            site: display url
            link: link to result
            preview: short snippet of site's content
        """
        results = list()
        res = self.service.cse().list(q=query,
                                      num=5,
                                      lr="lang_en",
                                      cx=engine_id
                                      ).execute()
        for result in res['items']:
            results.append({
                'title': result['title'],
                'site': result['displayLink'],
                'link': result['link'],
                'preview': result['snippet'],
            })
        return results


if __name__ == "__main__":
    cse = CSE()
    res = cse.search("qmind")
    for r in res:
        print("-"*80)
        pprint(r)
    print("-"*80)
