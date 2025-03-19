import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from instagrapi import Client
import os
import random

# Instagram credentials
USERNAME = "ahstros2018"
PASSWORD = "Walnut-Excess4-Emcee-Mouth"

# TinyLlama model details
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
)

def generate_caption(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the generated caption
    if "New caption:" in generated_text:
        generated_text = generated_text.split("New caption:")[-1].strip()
    
    return generated_text


# Path to the test image (Update when using a directory)
TEST_IMAGE_PATH = "C:/Users/orvin/post_automation/test_image.jpg"
IMAGE_FOLDER = None  # Change this to the folder path when using multiple images

def login_instagram():
    client = Client()
    client.login(USERNAME, PASSWORD)
    return client

def get_image():
    if IMAGE_FOLDER and os.path.isdir(IMAGE_FOLDER):
        images = [os.path.join(IMAGE_FOLDER, img) for img in os.listdir(IMAGE_FOLDER) if img.endswith((".jpg", ".png"))]
        if images:
            return random.choice(images)
    return TEST_IMAGE_PATH

def post_to_instagram(client, image_path, caption):
    client.photo_upload(image_path, caption)
    print(f"Posted: {image_path} with caption: {caption}")

if __name__ == "__main__":
    client = login_instagram()
    image_path = get_image()
    prompt = (
        "Generate a creative and engaging Instagram caption for a beauty salon's latest post. "
        "Use an uplifting and professional tone. Example captions: "
        "\"Glow like never before! ✨ Book your appointment today! 💇‍♀️ #BeautySalon\" "
        "\"Your transformation starts here! 🌸 Come visit us today. 💖 #SalonVibes\" "
        "\n\nNew caption:"
    )
    caption = generate_caption(prompt)
    post_to_instagram(client, image_path, caption)
