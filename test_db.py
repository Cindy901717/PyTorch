from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import os

writer = SummaryWriter("logs")
img_dir= "data/train/ants_image"
img_files = sorted(os.listdir(img_dir))

for step,file_name in enumerate(img_files):
    if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path = os.path.join(img_dir, file_name)
    img_PIL = Image.open(img_path)
    img_array = np.array(img_PIL)
    writer.add_image("test", img_array, step, dataformats='HWC')
# writer.add_image()
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()