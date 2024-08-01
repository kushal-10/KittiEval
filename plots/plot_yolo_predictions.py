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
        bbox, color = data
        # print(bbox, color)
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = bbox
        rect = patches.Rectangle((top_left_x, top_left_y),
                                 bottom_right_x - top_left_x,
                                 bottom_right_y - top_left_y,
                                 linewidth=2,
                                 edgecolor=color,  # Use specified color
                                 facecolor='none')
        ax.add_patch(rect)

    # Set axis off
    ax.axis('off')

    # Display the image with the bounding boxes
    plt.show()

def plot_predictions(image_data):
    """
    Plot predictions from yolo model
    :param image_data: A list of prediction metadata
    :return: None
    """

    for data in image_data:
        for k in list(data.keys()):
            image_path = k
            pred_data = data[k]
            plot_data = []
            for pred_item in pred_data:
                for item_key in pred_item:
                    bbox = pred_item[item_key]
                    if item_key == "Car":
                        color = 'b'
                    elif item_key == "Car_prediction":
                        color = 'g'
                        bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                    elif item_key == "DontCare":
                        color = 'r'

                    if len(bbox) < 4 or bbox[3] - bbox[1] <= 40:
                        print(bbox)
                    else:
                        plot_data.append((bbox, color))

            plot_bboxes_with_colors(image_path, plot_data)

        # break




if __name__ == '__main__':

    json_path = os.path.join("results", "easy", "jameslahm_yolov10x_test.json")

    with open(json_path, 'r') as f:
        image_data = json.load(f)

    # for data in image_data:
    #     for k in list(data.keys()):
    #         pred_data = data[k]
    #         for pred_item in pred_data:
    #             for item_key in pred_item:
    #                 if item_key == 'Car' and len(pred_item[item_key]) != 4:
    #                     print("LOL for Base Car")
    #                 elif item_key == 'DontCare' and len(pred_item[item_key]) != 4:
    #                     print("LOL for Base DontCare")
    #                 elif item_key == 'Car_prediction' and len(pred_item[item_key]) != 5:
    #                     print("LOL for Base Car_prediction")
    plot_predictions(image_data)


