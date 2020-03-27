import os
import requests

def download(fname: str):
    if not os.path.exists(fname):
        with open(fname, 'wb') as f:
            BaseURL = 'https://jackwish.net/storage/models/tests/'
            response = requests.get(BaseURL + fname)
            f.write(response.content)
        f.close()
    return fname
