import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from instagrapi import Client
import os
import random
import sys
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Instagram credentials from environment variables
USERNAME = os.getenv("IG_USERNAME")
PASSWORD = os.getenv("IG_PASSWORD")

# Check if credentials are loaded
if not USERNAME or not PASSWORD:
    raise ValueError("Instagram username or password not set in .env file!")

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


# Folder where images are stored (user must set this)
IMAGE_FOLDER = "C:/Users/orvin/post_automation/images"  # Change this to your folder path of images

# Default test image 
TEST_IMAGE_PATH = "C:/Users/orvin/post_automation/test_image.jpg"

def login_instagram():
    client = Client()
    client.load_settings("settings.json")
    client.login(USERNAME, PASSWORD)
    client.get_timeline_feed()
    return client

def get_image():
    """Retrieve an image from the specified folder or use a test image."""
    if IMAGE_FOLDER and os.path.isdir(IMAGE_FOLDER):
        images = [os.path.join(IMAGE_FOLDER, img) for img in os.listdir(IMAGE_FOLDER) if img.lower().endswith((".jpg", ".png"))]
        if images:
            return random.choice(images)  # Pick a random image
    print("No valid images found in the folder. Using test image instead.")
    return TEST_IMAGE_PATH  # Default test image

def post_to_instagram(client, image_path, caption):
    client.photo_upload(image_path, caption)
    print(f"Posted: {image_path} with caption: {caption}")

def post_story_to_instagram(client, image_path):
    """Upload an image as an Instagram Story."""
    client.photo_upload_to_story(image_path)
    print(f"Story posted: {image_path}")

if __name__ == "__main__":
    client = login_instagram()
    
    # Check if an argument was provided
    if len(sys.argv) < 2:
        print("Usage: python insta_post.py <1 for feed | 2 for story>")
        sys.exit(1)

    choice = sys.argv[1].strip()

    # Ensure the choice is valid
    if choice not in ["1", "2"]:
        print("Invalid choice. Use 1 for feed or 2 for story.")
        sys.exit(1)

    image_path = get_image()
    client.delay_range = [1, 3] #adds random delay between 4.5 - 5 mins

    if choice == "1":
        prompt = (
            "Generate a creative and engaging Instagram caption for a beauty salon's latest post. "
            "Use an uplifting and professional tone, don't include quotations in the quote. Example captions: "
            "\"Glow like never before! ✨ Book your appointment today! 💇‍♀️ #BeautySalon\" "
            "\"Your transformation starts here! 🌸 Come visit us today. 💖 #SalonVibes\" "
            "\n\nNew caption:"
        )
        caption = generate_caption(prompt)
        post_to_instagram(client, image_path, caption)

    elif choice == "2":
        post_story_to_instagram(client, image_path)
