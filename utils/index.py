import pdfminer
import uuid
 
def generate_random_string(length):
    random_string = str(uuid.uuid4())[:length]
    return random_string