import requests
from metaphone.metaphone import doublemetaphone


class PhoneticEncoding:

    def __init__(self):
        pass

    @staticmethod
    def metaphone3(s):
        return requests.post('http://localhost:8080/api/v1/metaphone', json={'input': s}).content.decode('utf-8')

    @staticmethod
    def metaphone2(s):
        return doublemetaphone(input=s)[0]