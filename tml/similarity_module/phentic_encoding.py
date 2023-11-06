import requests
from metaphone.metaphone import DoubleMetaphone


class PhoneticEncoding:

    def __init__(self):
        self.dm = DoubleMetaphone()

    def metaphone3(self, s):
        return requests.post('http://localhost:8080/api/v1/metaphone', json={'input': s}).content.decode('utf-8')

    def metaphone2(self, s):
        return self.dm.parse(input=s)[0]