import requests
from metaphone.metaphone import doublemetaphone


class PhoneticEncoding:

    def __init__(self):
        pass

    def metaphone3(self, s):
        return requests.post('http://localhost:8080/api/v1/metaphone', json={'input': s}).content.decode('utf-8')

    def metaphone2(self, s):
        return doublemetaphone(input=s)[0]