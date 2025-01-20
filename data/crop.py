import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

boundary_left = 60
boundary_right = 260
kernel_size = 25
half_size = kernel_size // 2

def bfs_find_largest_region(image):
    rows, cols = image.shape
    visited = np.zeros_like(image, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

    largest_region_size = 0
    largest_region_center = None

    def bfs(start):
        queue = deque([start])
        region_size = 0
        region_points = []

        while queue:
            x, y = queue.popleft()
            if visited[x, y]:
                continue
            visited[x, y] = True
            region_size += 1
            region_points.append((x, y))
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and image[nx, ny] == 255:
                    queue.append((nx, ny))
                    
        if region_size > 0:
            region_points = np.array(region_points)
            center = np.mean(region_points, axis=0).astype(int)
            return region_size, (center[0], center[1])
        else:
            return 0, None

    for i in range(rows):
        for j in range(boundary_left, boundary_right+1):
            if image[i, j] == 255 and not visited[i, j]:
                region_size, region_center = bfs((i, j))
                if region_size > largest_region_size:
                    largest_region_size = region_size
                    largest_region_center = region_center

    return largest_region_size, largest_region_center

def main(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (320, 200), interpolation=cv.INTER_AREA)

    _, thresh = cv.threshold(img, 40, 255, cv.THRESH_BINARY_INV)
    thresh[:, 0:60] = 0
    thresh[:, 260:320] = 0

    output = thresh.copy()
    rows, cols = output.shape
    counter = 0 

    neighbor_sum = np.zeros_like(output, dtype=int)

    for i in range(half_size, rows - half_size):
        for j in range(boundary_left + 1, boundary_right - half_size): 
            if (output[i,j] > 0):
                window_sum = np.sum(output[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1])
                neighbor_sum[i,j] = window_sum
                counter = counter + 1

    max_value = neighbor_sum.max()
    max_position = np.unravel_index(np.argmax(neighbor_sum), neighbor_sum.shape)
    print(max_position)
    center_x, center_y = max_position

    rect_width, rect_height = 225, 100
    x_start = max(center_x - rect_height // 2, 0)
    x_end = min(center_x + rect_height // 2, rows)
    y_start = max(center_y - rect_width // 2, 0)
    y_end = min(center_y + rect_width // 2, cols)

    actual_width = y_end - y_start
    actual_height = x_end - x_start

    print(f'Rectangle dimensions: {actual_width}x{actual_height}')
    cropped = img[x_start:x_end, y_start:y_end]
    print(cropped.shape)

    return cropped

if __name__ == '__main__':
    img_path = './openeds_sample/6400/000_SAC_left_0.29689724_0.23739485_0.92493011_0.02829531_0.00133197_0.00047214.png'
    main(img_path=img_path)