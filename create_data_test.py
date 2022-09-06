import cv2
import random
from datasets import HandwrittenDigitsDataset

save_folder = './Data/test/'

test_dataset = HandwrittenDigitsDataset(train=False)

s = 100
indexs = []
for i in range(s):
    index = random.randint(0, len(test_dataset))
    while index in indexs:
        index = random.randint(0, len(test_dataset))
    indexs.append(index)
    image, label = test_dataset[index]
    save_file_path = save_folder + f'image_{index}_{label}.jpg'
    cv2.imwrite(save_file_path, image)