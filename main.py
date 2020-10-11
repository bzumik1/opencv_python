import matplotlib.pyplot as plt

from SoftwareVision import ImageBGR, ImagePoint

print("STARTING")

image = ImageBGR.from_file("resources/test_svao_small01.png")

# IMAGE PROPERTIES
shape = image.shape
size = image.size
print("IMAGE PROPERTIES")
print(f"Image size is {shape[0]}x{shape[1]}x{shape[2]}, where {shape[2]} represents number of channels.")
print(f"Image width is {image.width} and height is {image.height}.")
print(f"Image takes {size/1000} kilobytes on your hard drive.")


# GRAYSCALE IMAGE
plt.imshow(image.gray(), cmap='gray')
plt.title("GRAYSCALE IMAGE")
plt.show()

# LAB
lab = image.lab()
plt.subplot(131)
plt.imshow(lab[:, :, 0], cmap='gray')
plt.title('L')
plt.subplot(132)
plt.imshow(lab[:, :, 1], cmap='RdYlGn_r')
plt.title('a')
plt.subplot(133)
plt.imshow(lab[:, :, 2], cmap='YlGnBu_r')
plt.title('b')
plt.show()

# RGB
plt.imshow(image.rgb())
plt.title("RGB IMAGE")
plt.show()

# BGR
plt.imshow(image.bgr())
plt.title("BGR IMAGE")  # just to demonstrate color channels are switched
plt.show()

# RESIZE
plt.imshow(image.resize(100, 200).rgb())
plt.title("RESIZED IMAGE")
plt.show()

# ROTATE
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(image.rotate(-45, True).rgb())
axs[0].set_title(f"ROTATION WITH SAME RATION")

axs[1].imshow(image.rotate(-45, False).rgb())
axs[1].set_title(f"ROTATION SHOWING WHOLE IMAGE")
plt.show()

# HISTOGRAM
plt.hist(image.gray().ravel(), 256, [0, 256])  #just for comparason
plt.plot(image.histogram(), color="r")
plt.title("HISTOGRAM OF GRAYSCALE IMAGE")
plt.show()

# SAVE IMAGE AS BMP
image.save_as_bmp("resources/", "test_bmp")

# MANUAL PERSPECTIVE TRANSFORMATION
corners = [ImagePoint(85, 126), ImagePoint(249, 63), ImagePoint(135, 249), ImagePoint(313, 164)]
plt.imshow(image.perspective_transform(corners).rgb())
plt.title("MANUAL PERSPECTIVE TRANSFORMATION")
plt.show()

# AUTODETECT CORNERS
plt.imshow(image.rgb())
for p in image.corners:
    plt.plot(p.x, p.y, 'bo')
plt.title("AUTODETECT CORNERS")
plt.show()

# MANUAL PERSPECTIVE TRANSFORMATION
paths = ["resources/test_svao_small01.png",
         "resources/test_svao_small02.png",
         "resources/test_svao_small03.png",
         "resources/test_svao_small04.png",
         "resources/test_svao_small05.png"]

images = [ImageBGR.from_file(p) for p in paths]

for i, image in enumerate(images):
    index = i+1
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(image.rgb())
    axs[0].set_title(f"ORIGINAL IMAGE {index}")

    axs[1].imshow(image.automatic_perspective_transform().rgb())
    axs[1].set_title(f"AUTOMATIC PERSPECTIVE TRANSFORMATION {index}")

    plt.show()

