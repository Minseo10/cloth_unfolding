import numpy as np
import cv2

# 예시로 사용할 이미지 배열
# image_array = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
#                         [0, 1, 0, 1, 1, 1, 0, 0],
#                         [0, 1, 0, 1, 1, 1, 0, 0],
#                         [0, 0, 0, 1, 1, 1, 0, 0],
#                         [0, 0, 0, 0, 0, 0, 0, 0]])
#
# image_array = np.array([[0, 0, 0, 1, 1, 1, 0, 0],
#                         [0, 0, 0, 1, 1, 1, 0, 0],
#                         [0, 0, 0, 1, 1, 1, 0, 0],
#                         [0, 0, 0, 1, 1, 1, 0, 0],
#                         [0, 0, 0, 1, 1, 1, 0, 0]])

image_array = np.load("/home/hjeong/code/unfolding_cloth/datasets/temp/mask.npy")

# 이미지 배열을 uint8로 변환하여 0과 255 값으로 바꿔줍니다.
image_array = (image_array * 255).astype(np.uint8)

# 이미지 배열에서 1로 채워진 부분의 좌표를 찾습니다.
contours, _ = cv2.findContours(image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bounding_box_area = [cv2.contourArea(contour) for contour in contours]
# 각 contour의 bounding box를 구합니다.
bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

max_area_index = np.argmax(bounding_box_area)
largest_bbox_coordinates = bounding_boxes[max_area_index]

print("Bounding boxes:", bounding_box_area)
print("Bounding boxes:", bounding_boxes)
print("largest_bbox_coordinates:", largest_bbox_coordinates)


# 원본 이미지에 가장 큰 bounding box를 그립니다.
image_with_bbox = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
x, y, w, h = largest_bbox_coordinates
cv2.rectangle(image_with_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 원본 이미지와 bounding box가 그려진 이미지를 함께 저장합니다.
cv2.imwrite("datasets/temp/original_with_bbox.jpg", image_with_bbox)