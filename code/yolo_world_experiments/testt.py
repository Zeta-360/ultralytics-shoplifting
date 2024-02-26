from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO('yolov8l-world.pt')  # or choose yolov8m/l-world.pt
# model = YOLO('yolov8s-world.pt')  # or choose yolov8m/l-world.pt

# Define custom classes
# model.set_classes(["person", "bus"])

# image is of person shopping in a store
# model.set_classes(["person", "shopping_cart", "store", "aisle", "shelf", "product", "price_tag", "checkout_counter", "cashier", "shopping_bag",])

# image is of person in computer lab
# model.set_classes(["person", "mouse", "keyboard", ])
model.conf = 0.000001
model.iou = 0.5

final_classes = [""]
# model.ckpt_path='yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pth'
# import torch
# state_dict = torch.load('yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pt', map_location=torch.device('cpu'))
# model.load_state_dict({'state_dict': state_dict['state_dict']})
# model.load_state_dict(state_dict['state_dict'])
# model.load_state_dict('yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pt')

# final_classes.extend(["mouse", "keyboard", "watch", "bottle", "laptop", "cell_phone", "chair", "table", "monitor", "keyboard", "mouse", "backpack", ])
# final_classes.extend(["hand", "mask"])
# final_classes.extend(["packets"])
# final_classes.extend(["chips packets"])
# final_classes.extend(["all goods on shelf"])
# final_classes.extend(["all black objects"])
# final_classes.extend(["chips packets"])
# final_classes.extend(["white objects"])
final_classes.extend(["person holding objects"])
# final_classes.extend(["bottle"])
# final_classes.extend(["object in person hand"])
# final_classes.extend(["girl wearing mask"])
# final_classes.extend(["person standing"])
print(final_classes)
model.set_classes(final_classes)


# model.save("custom_yolov8s.pt")
# model = YOLO('custom_yolov8s.pt')


# Execute prediction for specified categories on an image
# results = model.predict('data/imgs/man_picking.jpg')
# results = model.predict('data/imgs/girl_lifting.jpg')
video_complete_path = 'data/vids/man_picking.mp4'
results = model.predict(video_complete_path, stream=True)
video_name = video_complete_path.split('/')[-1].split('.')[0]

output_base_dir = '/opt/homebrew/runs/detect'
import os
os.listdir(output_base_dir)
# 'predict6', 'predict', 'predict3', 'predict4'
# find last index in directory list names

import time
epoch_time = int(time.time())
output_dir = os.path.join(output_base_dir, f'{video_name}_{epoch_time}')
output_dir_frames = os.path.join(output_dir, 'frames')
os.mkdir(output_dir)
for i,result in enumerate(results):
    final_frame_path = os.path.join(output_dir_frames, f'{i}.jpg')
    result.show()
    result.save(final_frame_path)
# results.save(output_dir)

# Show results
# results[0].show()