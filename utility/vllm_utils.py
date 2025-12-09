from base64 import b64encode
from ollama_utils import OllamaUtils
from typing import List
from PIL import Image
from io import BytesIO


class VLLMUtils:

    def __init__(self):
        ollama_utils = OllamaUtils()

    def polish_prompt_en(self, original_prompt: str, images: List[Image.Image]):
        prompt = f"""
            # Edit Instruction Rewriter
            You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  

            Please strictly follow the rewriting rules below:

            ## 1. General Principles
            - Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.  
            - If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
            - Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
            - All added objects or modifications must align with the logic and style of the edited input image’s overall scene.  

            ## 2. Task Type Handling Rules
            ### 1. Add, Delete, Replace Tasks
            - If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
            - If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
                > Original: "Add an animal"  
                > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
            - Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
            - For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  

            ### 2. Text Editing Tasks
            - All text content must be enclosed in English double quotes `" "`. Do not translate or alter the original language of the text, and do not change the capitalization.  
            - **For text replacement tasks, always use the fixed template:**
                - `Replace "xx" to "yy"`.  
                - `Replace the xx bounding box to "yy"`.  
            - If the user does not specify text content, infer and add concise text based on the instruction and the input image’s context. For example:  
                > Original: "Add a line of text" (poster)  
                > Rewritten: "Add text \"LIMITED EDITION\" at the top center with slight shadow"  
            - Specify text position, color, and layout in a concise way.  

            ### 3. Human Editing Tasks
            - Maintain the person’s core visual consistency (ethnicity, gender, age, hairstyle, expression, outfit, etc.).  
            - If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.  
            - **For expression changes, they must be natural and subtle, never exaggerated.**  
            - If deletion is not specifically emphasized, the most important subject in the original image (e.g., a person, an animal) should be preserved.
                - For background change tasks, emphasize maintaining subject consistency at first.  
            - Example:  
                > Original: "Change the person’s hat"  
                > Rewritten: "Replace the man’s hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged"  

            ### 4. Style Transformation or Enhancement Tasks
            - If a style is specified, describe it concisely with key visual traits. For example:  
                > Original: "Disco style"  
                > Rewritten: "1970s disco: flashing lights, disco ball, mirrored walls, colorful tones"  
            - If the instruction says "use reference style" or "keep current style," analyze the input image, extract main features (color, composition, texture, lighting, art style), and integrate them into the prompt.  
            - **For coloring tasks, including restoring old photos, always use the fixed template:** "Restore old photograph, remove scratches, reduce noise, enhance details, high resolution, realistic, natural skin tones, clear facial features, no distortion, vintage photo restoration"  
            - If there are other changes, place the style description at the end.

            ## 3. Rationality and Logic Checks
            - Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" should be logically corrected.  
            - Add missing key information: if position is unspecified, choose a reasonable area based on composition (near subject, empty space, center/edges).  

            # Output Format Example
            "211 floors high skyscrape, majesticly dominating the crowded street below..."

            Prompt to be rewritten: {original_prompt}
        """
        return self.interact(prompt, images)
    
    def make_caption(self, image: Image.Image):
        prompt = """You are a bot capable of describing a given image in a detailed manner.
        Given the following image, please describe it, with particular care for its setting.

        Respond exclusively with the following structure:

        {
        "description": string
        }

        Here is an example of response:

        {
        "description": "A solemn priest stands in front of a congregation, inside of a gothic church, in the warm morning light filtering from coloured windows."
        }

        Description field must contain a string of no more than 30 words

        Do not add any further comment, nor markdown notation.

        """
        return self.interact(prompt, [image])

    def inclusion_evaluation(self, items: List[str], image: Image.Image):
        prompt = f"""You are a bot capable of deciding whether the given image contains every stated elements or not. Given a list of stated elements:

        {items}

        Respond strictly with the following structure:

        {{
            "response": boolean,
            "missing": []
        }}

        Here are some examples of responses:

        {{
            "response": true,
            "missing": []
        }}

        {{
            "response": false,
            "missing": ["a dog", "a cradle"]
        }}

        Do not use true as a value for response when there are missing items, nor use false where there are not missing items. Examples of unallowed responses:

        {{
            "response": true,
            "missing": ["a pidgeon", "a car"]
        }}

        {{
            "response": false,
            "missing": []
        }}

        Do not add any further comment, nor markdown notation, nor backticks, nor the word "json". Here is an example of unwanted response:
        ```json
            {{
            "response": false,
            "missing": ["A lighthouse"]
            }}
        ```
        While this is an example of good response:
        {{
            "response": false,
            "missing": ["A lighthouse"]
        }}

        """

        return self.interact(self, prompt, [image])

    def make_caption_env(self, items, description):
        prompt = f"""You are a bot capable of images captioning. For each item in the following list:

        {items}

        suggest a caption that describes that concept, as a subject of an image, blending in a setting described as follows:

        '{description}'

        Respond exclusively with the following structure:

        {{
            "captions": []
        }}

        so that each item of the list corresponds with an item of the array inside the "captions" field.

        Here are an example of response for a two item list, based on the description "A fun playground full of children, with sandboxes and swinging horses":

        {{
            "captions": [
            "A playful grizzly bear build sandcastles in a playground full of children",
            "A classroom taking lessons of fun and joy in a fun playground"
            ]
        }}

        Do not add any further comment, nor markdown notation, nor backticks, nor the word "json". Here is an example of wrong response:
        ```json
            {{
            "response": false,
            "missing": ["A lighthouse"]
            }}
        ```
        While this is an example of good response:
        {{
            "response": false,
            "missing": ["A lighthouse"]
        }}

        """
        return self.interact(self, prompt)

    def interact(self, prompt: str, images: List[Image.Image]):
        base64_images = []
        for image in images:
            with BytesIO() as buffered:
                image.save(buffered, format="JPEG")
                base64_images.append(b64encode(buffered.getvalue()).decode("utf8"))
        payload = {"role": "user", "content": prompt, "images": base64_images}
        return self.ollama_utils.interact(payload)
