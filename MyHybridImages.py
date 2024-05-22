import math
import numpy as np

from MyConvolution import convolve


def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.
        :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour
        shape=(rows,cols,channels))
        :type numpy.ndarray
        :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
        :type float
        :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
        :type numpy.ndarray
        :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage
        before subtraction to create the high-pass filtered image
        :type float
    :returns returns the hybrid image created
    by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with
    a high-pass image created by subtracting highImage from highImage convolved with
    a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
        :rtype numpy.ndarray
    """
    # Process and normalize input images prior to convolution
    lowImage, highImage = resize_images(lowImage, highImage)
    highImage = highImage.astype(np.float32) / 255.0
    lowImage = lowImage.astype(np.float32) / 255.0
    
    low_pass_filter = makeGaussianKernel(lowSigma)
    lpass_lowImage = convolve(lowImage, low_pass_filter) # Perform low pass filtering on lowImage

    high_pass_filter = makeGaussianKernel(highSigma)
    lpass_highImage = convolve(highImage, high_pass_filter) # Perform low pass filtering on highImage

    highfImage = highImage - lpass_highImage # High pass image
    hybrid_image = lpass_lowImage + highfImage

    # Clip the combined image and rescale to greyscale/coloured image
    clipped_image = np.where(hybrid_image < 0, 0, hybrid_image)
    clipped_image = np.where(clipped_image > 1, 1, clipped_image)
    return (clipped_image * 255).astype(np.uint8)

def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or floor(8*sigma+1)+1
    (whichever is odd) as per the assignment specification.
        :param sigma: the standard deviation used to calculate size and values in kernel
    
    :returns The kernel
    :rtype numpy.ndarray
    """
    size = math.floor(8.0 * sigma + 1.0)
    size += 1 if size % 2 == 0 else 0
    half_width = (size - 1) / 2
    ax = np.arange(-half_width, half_width + 1)
    # Extract the coordinates for gaussian formula
    x_pos = np.repeat(ax[:, None], ax.shape[0], axis=1)
    y_pos = np.repeat(ax[None, :], ax.shape[0], axis=0)

    # 2D Gaussian equation with np to vectorize approach
    constant = 1 / (2 * math.pi * (sigma ** 2))
    kernel = constant * np.exp(-0.5 * (x_pos**2 + y_pos**2) / sigma**2)
    return kernel / np.sum(kernel) # Normalize the kernel so that it sums to 1

def resize_images(image1, image2):
  """
  Pads two images to the same size
    :param image1: A numpy array representing the first image.
    :param image2: A numpy array representing the second image.

  :returns A tuple of two numpy arrays, both of which are padded to the same size.
  """
  # Get the dimensions of the two images.
  image1_height, image1_width, image1_channels = image1.shape
  image2_height, image2_width, image2_channels = image2.shape

  if (image1.shape[:2] == image2.shape[:2]):
    if (image1_channels == image2_channels):
      return image1, image2
    else:
      if (image1_channels < image2_channels):
        image1 = np.repeat(image1[:, :, np.newaxis], 3, axis=2)
      else:
        image2 = np.repeat(image2[:, :, np.newaxis], 3, axis=2)
      return image1, image2

  # Determine the desired dimensions of the padded arrays.
  desired_width = max(image1_width, image2_width)
  desired_height = max(image1_height, image2_height)
  desired_channel = max(image1_channels, image2_channels)

  # Create two new empty arrays to store the padded images.
  if desired_channel == 3:
    padded_image1 = np.empty((desired_height, desired_width, 3), dtype=np.uint8)
    padded_image2 = np.empty((desired_height, desired_width, 3), dtype=np.uint8)
  elif desired_channel == 1:
    padded_image1 = np.empty((desired_height, desired_width), dtype=np.uint8)
    padded_image2 = np.empty((desired_height, desired_width), dtype=np.uint8)
  else:
    raise ValueError("Images must have either 1 or 3 channels.")
    
  for y in range(desired_height):
    for x in range(desired_width):
      # If the coordinates are within the bounds of the first image, copy the pixel from the first image.
      if y < image1_height and x < image1_width:
        padded_image1[y, x] = image1[y, x]
      # If the coordinates are within the bounds of the second image, copy the pixel from the second image.
      if y < image2_height and x < image2_width:
        padded_image2[y, x] = image2[y, x]
      # Otherwise, set the pixel to black.
      else:
        padded_image1[y, x] = [0, 0, 0]
        padded_image2[y, x] = [0, 0, 0]

  return padded_image1, padded_image2
