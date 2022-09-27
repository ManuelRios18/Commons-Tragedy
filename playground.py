import os
import cv2
from utils import load_pickle


def load_dream(d_id):
    root_path = f"/home/manuel/Documents/Research/Commons-Tragedy/logs/dreamerv2_ma_2p/dreams/player_0/data"
    return load_pickle(os.path.join(root_path, f"dream_{d_id}.pickle"))


player_num = 0
dream_id = 31
frame_num = 0


data = load_dream(dream_id)
print("image", frame_num)
while True:
    frame = data[frame_num][:, :, ::-1]
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('d') and frame_num < len(data) - 1:
        frame_num += 1
        print("image", frame_num)
    if key & 0xFF == ord('a') and frame_num > 0:
        frame_num -= 1
        print("image", frame_num)
    if key & 0xFF == ord('p'):
        print("Saving frame", frame_num, " of dream", dream_id)
        cv2.imwrite(f"logs/dreams/dream_{dream_id}_frame_{frame_num}.png", frame)
    if key & 0xFF == ord('+'):
        dream_id += 1
        frame_num = 0
        data = load_dream(dream_id)
        print("switching to dream: ", dream_id)
    if key & 0xFF == ord('-'):
        dream_id -= 1
        frame_num = 0
        data = load_dream(dream_id)
        print("switching to dream: ", dream_id)

