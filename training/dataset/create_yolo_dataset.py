# File to convert the data-set into YOLO compatible format,
# as mentioned here - https://docs.ultralytics.com/datasets/detect/

from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import os

dataset = load_dataset("Koshti10/omegalol1")

if not os.path.exists(os.path.join('data', 'images')):
    os.makedirs(os.path.join('data', 'images'))

output_folder = os.path.join('data', 'images')

print(len(dataset['train']['image']))

#
# counter = 0
# # Iterate through the dataset
# for idx, example in enumerate(dataset):
#     # Replace 'image' with the key that holds the image data
#
#     print(example[0])
#     # image_url = example['image']
#     #
#     # # Download the image
#     # response = requests.get(image_url)
#     # img = Image.open(BytesIO(response.content))
#     #
#     # # Define the path to save the image
#     # image_path = os.path.join(output_folder, f'image_{idx}.png')
#     #
#     # # Save the image
#     # img.save(image_path)
#     #
#     # print(f'Saved image {idx} to {image_path}')
#
#     counter += 1
#     if counter == 20:
#         break

print("SAVED ALL, LFG")
