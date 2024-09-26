#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install transformers torch Pillow gradio


# In[3]:


import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load pre-trained T5-small model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# In[6]:


from PIL import Image
import numpy as np

def process_image(image_path):
    # Load image
    image = Image.open(image_path)
    
    # Preprocess image
    image = image.convert('RGB')
    image = np.array(image)
    
    # Perform OCR
    inputs = tokenizer(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    extracted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return extracted_text


# In[7]:


import gradio as gr

def search_text(extracted_text, keyword):
    # Perform keyword search
    search_results = [line for line in extracted_text.split('\n') if keyword in line]
    
    return search_results

def web_application():
    # Create interface
    interface = gr.Interface(
        fn=lambda image, keyword: (process_image(image), search_text(process_image(image), keyword)),
        inputs=["image", "text"],
        outputs=["text", "text"],
        title="OCR Web Application",
        description="Upload an image and search for keywords in the extracted text."
    )
    
    return interface

if __name__ == "__main__":
    web_application().launch()


# In[12]:


{
  "name": "ocr-web-app",
  "description": "OCR Web Application",
  "template": "gradio"
}


# In[ ]:




