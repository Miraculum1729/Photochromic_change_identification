import numpy as np
import cv2
import glob


# Load all images, and so on
img_paths = glob.glob('./imgs/*.*')
if not img_paths:
    raise ValueError("No images found. Please make sure there are image files in the ./imgs/ directory.")
# Ensure image paths are sorted alphabetically
img_paths.sort()
# Ensure there are at least two images
if len(img_paths) < 2:
    raise ValueError("At least two images are required for comparison.")
# Read images
imgs = [cv2.imread(img_path) for img_path in img_paths]

def crop_image_to_square(img, visualize=False):
    """
    Detect two green calibration points in the image (top-left and bottom-right)
    and crop the image into a square.
    
    :param img: Input image in BGR format.
    :return: Cropped square image.
    """
    if img is None:
        print("Error: Input image is empty.")
        return None

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV range for green color
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 2:
        print("Error: Not enough green calibration points detected.")
        return None
    
    # Assume the two largest contours correspond to the top-left and bottom-right calibration points
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    top_left = contours[0]
    bottom_right = contours[1]

    # Get coordinates of calibration points
    x1, y1 = top_left[0][0]
    x2, y2 = bottom_right[0][0]
    # Calculate cropping boundaries
    new_x1 = min(x1, x2)
    new_x2 = max(x1, x2)
    new_y1 = min(y1, y2)
    new_y2 = max(y1, y2)

    # Crop image
    cropped_img = img[new_y1:new_y2, new_x1:new_x2]

    if visualize:
        # Display the cropped image
        cv2.imshow("Cropped Image", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped_img

def create_and_show_hole_mask(img, visualize=False):
    """
    Detect all circular holes in the image, create a mask containing these holes,
    and display the result.

    :param img: Input image in BGR format.
    :return: None
    """
    if img is None:
        print("Error: Input image is empty.")
        return

    # --- Step 1: Image preprocessing ---
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve circle detection accuracy
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)

    # --- Step 2: Detect circular holes using Hough Circle Transform ---
    
    # Note: These parameters may need adjustment depending on the images
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,           # Accumulator resolution, typically between 1 and 2
        minDist=110,      # Minimum distance between circle centers
        param1=140,       # High threshold for Canny edge detection
        param2=20,        # Accumulator threshold for circle detection
        minRadius=60,     # Minimum radius
        maxRadius=150     # Maximum radius
    )

    # --- Step 3: Create and display mask ---

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define HSV range for purple color
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([170, 255, 255])
    # Create purple mask
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    purple_holes = []

    # Create a black mask with the same size as the original image
    mask = np.zeros(img.shape[:2], dtype="uint8")

    # Check if any circles are detected
    if circles is not None:
        # Convert circle coordinates and radii to integers
        circles = np.round(circles[0, :]).astype("int")
        
        print(f"Successfully detected {len(circles)} circular holes.")

        # Iterate through all detected circles
        for (x, y, r) in circles:
            mask_single_circle = np.zeros(img.shape[:2], dtype="uint8")
            # Check whether the detected circle contains purple regions
            cv2.circle(mask_single_circle, (x, y), r, 255, -1)  # -1 means filled circle
            # If the circular region overlaps with the purple mask, it is considered purple
            if cv2.countNonZero(cv2.bitwise_and(mask_single_circle, purple_mask)) > 0:
                print(f"Purple circular hole detected at position: ({x}, {y}), radius: {r}")
                # Draw the detected purple hole on the original image
                cv2.circle(img, (x, y), r, (255, 0, 255), 2)
            # Fill the circular region as white on the black mask
            cv2.circle(mask, (x, y), r, 255, -1)

            # There are 9 hole regions arranged in a 3×3 grid,
            # numbered from 1 to 9 from left to right and top to bottom
            # Assign an index number to each detected hole
            hole_number = (y // (img.shape[0] // 3)) * 3 + (x // (img.shape[1] // 3)) + 1
            purple_holes.append(int(hole_number))
        print(f"Detected purple hole indices: {purple_holes}")
    else:
        print("No circular holes detected in the image.")

    if visualize:
        # Display the final image with detected holes
        cv2.imshow("All Circular Holes", img)

        print("Mask window displayed. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return purple_holes, img

# There are 9 hole regions arranged in a 3×3 grid,
# numbered from 1 to 9 from left to right and top to bottom.
# Process one image at a time and identify which hole region
# changes to purple compared with the previous image.

# Process the first image and identify purple regions
first_img = imgs[0]
crop_image = crop_image_to_square(first_img, visualize=False)
purple_holes_first, _ = create_and_show_hole_mask(crop_image, visualize=True)

# The sequence of purple hole indices represents a pattern
# similar to a smartphone unlock password
password = [purple_holes_first[0]]  # Initialize password list with the first detected hole

# Process subsequent images and compare changes
for i in range(1, len(imgs)):
    current_img = imgs[i]
    crop_image = crop_image_to_square(current_img, visualize=False)
    purple_holes_current, img_current = create_and_show_hole_mask(
        crop_image, visualize=True if i < len(imgs) - 1 else True
    )

    # Compare purple hole indices between current and previous image
    changed_holes = set(purple_holes_current) - set(purple_holes_first)
    if changed_holes:
        # If changes are detected, append them to the password list
        password.extend(changed_holes)
        print(f"In image {i+1}, the following hole regions changed: {changed_holes}")
    else:
        print(f"In image {i+1}, no hole region changes were detected.")

    # Update previous hole indices
    purple_holes_first = purple_holes_current

# Output the indices of the detected purple pores,
# representing the photochromic changes identified in each image.
print(f"Final identified password based on purple pore indices: {password}")

# Display the password with arrows indicating the photochromic sequence
print("Password corresponding to the photochromic sequence:",
      '->'.join(map(str, password)))

# Predefined correct password
correct_password = [1, 2, 3, 5, 9, 8, 7]

# Verify the detected password
if password == correct_password:
    print("Password correct")
else:
    print("Password incorrect")
