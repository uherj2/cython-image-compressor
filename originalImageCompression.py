from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.widgets import Slider, Button


class imageCompressor:
    def __init__(self, selected_image):
        self.selected_image = selected_image
        print(selected_image)

        # original image stores the original image
        # uses PIL library to load and convert image to numpy array
        self.original_image = np.array(Image.open(self.selected_image))


        # a dictionary that maps each k value to its corresponding compressed image (np array)
        self.kmeans_images = defaultdict(lambda: None)

    def initialize_centroids(self, data, k):
        return data[np.random.choice(data.shape[0], k, replace=False)]

    def assign_clusters(self, data, centroids):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, data, labels, k):
        return np.array([data[labels == i].mean(axis=0) for i in range(k)])

    def apply_label(self, label, centroids):
        print(centroids[label])
        return centroids[label]

    # method that accepts a k and preforms k means
    # result stored in k means images
    # new image is stored every time
    def kmeans(self, num):
        MAX_ITERS = 100
        k = num
        # flatten image
        data = self.original_image.reshape(-1, self.original_image.shape[2])

        # centroids
        centroids = self.initialize_centroids(data, k)
        print(centroids)

        for i in range(MAX_ITERS):
            # assign clusters
            label = self.assign_clusters(data, centroids)
            print(f'label: \n{label}')
            # update k means
            new_centroids = self.update_centroids(data, label, k)
            # optimize by check if stable
            if np.all(np.abs(new_centroids - centroids) < 1):
                print('state is stable')
                break

            centroids = new_centroids


        new_image = self.apply_label(label, centroids)
        new_image = new_image.astype(np.uint8)
        new_image = new_image.reshape(self.original_image.shape)

        self.kmeans_images[k] = new_image
        print(self.kmeans_images[k])
        imag = Image.fromarray(new_image)
        # imag.show()

    # returns either the original image (k = 0) or compressed k means image
    # retrieves image
    def get_image(self, k):
        if k == 0:
            return self.original_image

        if self.kmeans_images[k] is None:
            print('location is none')
            self.kmeans(k)

        return self.kmeans_images[k]


global current_kvalue

# List of image filenames
image_title_list = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg"]
image_compressor_list = []
for image in image_title_list:
    current_image = Image.open(image)
    image_array = np.array(current_image) / 255.0
    compressor_object = imageCompressor(image)
    image_compressor_list.append(compressor_object)

# Load the initial image
current_image_index = 0
initialImage = imageCompressor("test1.jpg")
image = Image.open(image_title_list[current_image_index])
image_array = np.array(image) / 255.0  # Normalize to range [0, 1]

# Create the initial plot
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.3)  # Adjust space for buttons and slider
ax.set_title("compression Adjustment")
ax.axis('off')

# Display the initial image
img_display = ax.imshow(image_array)

# Add a slider for brightness control
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])  # x, y, width, height
slider = Slider(ax_slider, 'Compression', 0, 5, valinit=0, valstep=1)

# Update function for slider
def update(val):
    scale = slider.val
    # current_kvalue = slider.val
    # Scale the brightness of the image
    # adjusted_image = np.clip(image_array * scale, 0, 5)  # Keep values in range [0, 1]
    adjusted_image = image_compressor_list[current_image_index].get_image(slider.val)
    img_display.set_data(adjusted_image)  # Update the displayed image
    fig.canvas.draw_idle()  # Redraw the canvas

# Function to load and display the next/previous image
def change_image(direction):
    global current_image_index, image_array
    if direction == 'next':
        # Move to the next image, wrapping around if necessary
        current_image_index = (current_image_index + 1) % len(image_title_list)
    elif direction == 'prev':
        # Move to the previous image, wrapping around if necessary
        current_image_index = (current_image_index - 1) % len(image_title_list)

    # Load the new image and update the display
    placeholder = slider.val
    print(placeholder)
    # curImage = Image.open(image_title_list[current_image_index])
    # image_array = np.array(curImage) / 255.0  # Normalize to range [0, 1]
    image_array = image_compressor_list[current_image_index].get_image(slider.val)
    img_display.set_data(image_array)  # Update the displayed image
    fig.canvas.draw_idle()  # Redraw the canvas

# Add "Previous" and "Next" buttons
ax_prev = plt.axes([0.1, 0.01, 0.1, 0.075])  # x, y, width, height
ax_next = plt.axes([0.8, 0.01, 0.1, 0.075])

button_prev = Button(ax_prev, 'Previous')
button_next = Button(ax_next, 'Next')

# Define the button callbacks
button_prev.on_clicked(lambda event: change_image('prev'))
button_next.on_clicked(lambda event: change_image('next'))

# Connect the slider to the update function
slider.on_changed(update)

# Show the plot
plt.show()
