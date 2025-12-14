import chromadb
import base64
import io
import json
from PIL import Image
from matplotlib import image as mpimg
from typing import List, Dict

class VectorStoreUtils: 

    def __init__(self):
        self.chromaClient = chromadb.EphemeralClient() 
        self.vllm = None
        self.collection = self.chromaClient.create_collection(name="eustachian_collection")
        self.lastId = 0
        
    def addToCollection(self, documents: List): 
        for i in documents:
         entry =  f"""
        {{ 
            "caption": "{i['caption']}",
            "base64": "{self._encode_image(i['base64'])}"
        }}
        """
        self.collection.add(
           ids = [str(self.lastId)],
           documents = [
              entry 
           ]
        )
        self.lastId += 1    

    def query(self, queries: List[str], n_results=5) -> Dict:
        results = [self._single_query(q, n_results) for q in queries]

        return results
    
    def _single_query(self, query_text: str, n_results: int) -> Image.Image:
        result = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        enc_retrieved = json.loads(result['documents'][0][0].replace("'", '"'))['base64']
        cap_retrieved = json.loads(result['documents'][0][0].replace("'", '"'))['caption']
        return {'img': self._decode_image(enc_retrieved), 'cap': cap_retrieved}

    def _encode_image(self, image_path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _decode_image(self, base64_img) -> Image.Image:
        dec_img = base64.b64decode(base64_img)
        bytez = io.BytesIO(dec_img)
        return Image.open(bytez)