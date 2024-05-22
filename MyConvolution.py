import numpy as np

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray: 
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders
        :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
        (the image is normalized when passed as input)
        :type numpy.ndarray
        :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
        :type numpy.ndarray
    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    kernel = np.array(kernel[::-1, ::-1]) # Flip kernel for convolution
    image_rows, image_cols = image.shape[:2]
    kernel_rows, kernel_cols = kernel.shape
    padded_image, channels = pad_image(image, kernel)

    convolved_image = np.zeros(image.shape)
    for ch in range(channels): # for each RBG and gray scale channels
        for x in range(0, image_cols):
            for y in range(0, image_rows):
                # Perform element-wise multiplication and sum up values to generate convolved pixel value
                if channels == 1:
                    convolved_image[y, x] = (kernel * padded_image[y:y+kernel_rows, x:x+kernel_cols]).sum()
                else:
                    convolved_image[y, x, ch] = (kernel * padded_image[y:y+kernel_rows, x:x+kernel_cols, ch]).sum()

    return convolved_image

def pad_image(image: np.ndarray, kernel: np.ndarray):
    """
    Pad an image to achieve zero-padding of the image to handle the borders in convolution
        :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
        :type numpy.ndarray
        :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
        :type numpy.ndarray
    :returns a tuple containing the padded image and the number of channels
    :rtype (numpy.ndarray,int)
    """
    kernel_rows, kernel_cols = kernel.shape
    tr = (kernel_rows - 1) // 2
    tc = (kernel_cols - 1) // 2
    if image.ndim == 2: # Pad each greyscale image appropriately
        padded_image = np.zeros((image.shape[0] + 2 * tr, image.shape[1] + 2 * tc), dtype=image.dtype)
        padded_image[tr:tr + image.shape[0], tc:tc + image.shape[1]] = image
    else:
        padded_image = np.zeros((image.shape[0] + 2 * tr, image.shape[1] + 2 * tc, image.shape[2]), dtype=image.dtype)
        padded_image[tr:tr + image.shape[0], tc:tc + image.shape[1], :] = image
    
    channels = 1 if (image.ndim == 2) else 3
    return padded_image, channels
