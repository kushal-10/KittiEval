import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_bboxes_with_colors(image_path, plot_data):
    # Open the image file
    image = Image.open(image_path)

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Iterate over each item in plot_data
    for data in plot_data:
        # print(bbox, color)
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = data
        rect = patches.Rectangle((top_left_x, top_left_y),
                                 bottom_right_x - top_left_x,
                                 bottom_right_y - top_left_y,
                                 linewidth=2,
                                 edgecolor='b',
                                 facecolor='none')
        ax.add_patch(rect)

    # Set axis off
    ax.axis('off')

    # Display the image with the bounding boxes
    plt.show()


if __name__ == '__main__':
    LABEL_DIR = os.path.join('data', 'labels')
    label_txt_files = os.listdir(LABEL_DIR)

    for txt_file in label_txt_files:
        txt_file_path = os.path.join(LABEL_DIR, txt_file)
        image_number = txt_file.split(".")[0]
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()

        image_number = str(image_number).zfill(6)
        image_path = os.path.join("data", "images", image_number + ".png")
        plot_data = []
        for line in lines:
            values = line.strip().split()
            if values[0] == 'Car':
                if float(values[7]) - float(values[5]) > 40:
                    plot_data.append([float(values[4]), float(values[5]), float(values[6]), float(values[7])])

        plot_bboxes_with_colors(image_path, plot_data)
