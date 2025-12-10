import chromadb
import base64
import io
from PIL import Image
from matplotlib import image as mpimg
from typing import List

class VectorStoreUtils: 

    def __init__(self):
        self.chromaClient = chromadb.EphemeralClient() 
        self.collection = self.chromaClient.create_collection(name="eustachian_collection")
        self.lastId = 0
        
    def addToCollection(self, documents: List): 
        for i in documents:
         entry =  f"""
        {{ 
            "caption": "{i.caption}",
            "base64": "{self._encodeImage(f'{i.image}')}"
        }}
        """
        self.collection.add(
           ids = [self.lastId],
           documents = [
              entry 
           ]
        )
        self.lastId += 1    

    def query(self, query_text: str, n_results=1):
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

    def _encode_image(self, image_path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _decode_image(self, base64_img, img_format: str) -> Image.Image:
        decoded_bytes = io.BytesIO(base64.b64decode(base64_img))
        decoded_image = mpimg.imread(decoded_bytes, format=img_format)

        return Image.fromarray(decoded_image)