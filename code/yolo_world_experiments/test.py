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
# model.ckpt_path='yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pth'
import torch
state_dict = torch.load('yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pt', map_location=torch.device('cpu'))
# model.load_state_dict({'state_dict': state_dict['state_dict']})
model.load_state_dict(state_dict['state_dict'])
# model.load_state_dict('yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pt')

# model.set_classes(["mouse", "keyboard", "watch", "bottle", "laptop", "cell_phone", "chair", "table", "monitor", "keyboard", "mouse", "backpack", ])
# model.set_classes(["hand", "mask"])
# model.set_classes(["packets"])
# model.set_classes(["white objects"])
model.set_classes(["person holding objects"])
# model.set_classes(["bottle"])
# model.set_classes(["girl wearing mask"])
# model.set_classes(["person standing"])


# model.save("custom_yolov8s.pt")
# model = YOLO('custom_yolov8s.pt')


# Execute prediction for specified categories on an image
# results = model.predict('data/imgs/man_picking.jpg')
results = model.predict('data/imgs/girl_lifting.jpg')

# Show results
results[0].show()