import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image.astype(np.uint8))

class BBoxSelect:
    def __init__(self):
        self.fig = None

    def run_gui(self, img, obj_name_list=[]):
        self.fig, self.ax = plt.subplots()

        self.key_call = self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.rectangle_call = RectangleSelector(self.ax, self.on_select, useblit=True,
                                  button=[1],  # Only left mouse button
                                  minspanx=5, minspany=5,  # Minimum span to make selection
                                  spancoords='pixels',  # Use pixels for span size
                                  interactive=False)

        self.image = img.copy()
        self.obj_name_list = obj_name_list

        self.refresh_img_status()

        self.ax.imshow(self.image.astype(np.uint8))
        self.ax.set_title(self.obj_name_list[self.obj_idx])
        plt.show()

        return self.obj_mask_dict

    def refresh_obj_status(self):
        # object level
        self.points = []
        self.current_mask = None

    def refresh_img_status(self):
        # image level
        self.obj_mask_dict = {}
        self.obj_idx = 0
        # object level
        self.refresh_obj_status()

    def close(self):
        print("close")
        if self.fig is not None:
            self.fig.canvas.mpl_disconnect(self.key_call)
            plt.close()
    
    def visualize_masks(self):
        self.ax.cla()
        self.ax.imshow(self.image.astype(np.uint8))
        for obj_name, obj_mask in self.obj_mask_dict.items():
            show_mask(obj_mask, self.ax)
        plt.show(block=True)

    def on_select(self, eclick, erelease):
        start_x, start_y = eclick.xdata, eclick.ydata  # Start coordinates
        end_x, end_y = erelease.xdata, erelease.ydata  # End coordinates
        x_min, x_max = int(np.min([start_x, end_x])), int(np.max([start_x, end_x]))
        y_min, y_max = int(np.min([start_y, end_y])), int(np.max([start_y, end_y]))
        self.points = [(x_min, y_min), (x_max, y_max)]

        self.current_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.current_mask[y_min:y_max, x_min:x_max] = 255

        self.ax.imshow(self.image.astype(np.uint8))
        show_mask(self.current_mask, self.ax, random_color=False)
        self.ax.set_title(self.obj_name_list[self.obj_idx])
        plt.draw()

    def on_key_press(self, event):
        if event.key == "escape":
            self.close()

        if event.key == " ": # cache current selected mask and wait for the next mask in the same image
            if self.current_mask is not None:
                self.obj_mask_dict[self.obj_name_list[self.obj_idx]] = self.current_mask
                self.obj_idx += 1

                self.refresh_obj_status()

                if self.obj_idx < len(self.obj_name_list):
                    self.ax.cla()
                    self.ax.imshow(self.image.astype(np.uint8))
                    self.ax.set_title(self.obj_name_list[self.obj_idx])
                    plt.show()
                else:
                    print("[done] number of cached masks:", len(self.obj_mask_dict))
                    self.visualize_masks()

        if event.key.lower() == "n": # skip current object and go to the [n]ext one
            self.obj_idx += 1

            self.refresh_obj_status()

            if self.obj_idx < len(self.obj_name_list):
                self.ax.cla()
                self.ax.imshow(self.image.astype(np.uint8))
                self.ax.set_title(self.obj_name_list[self.obj_idx])
                plt.show()
            else:
                print("[done] number of cached masks:", len(self.obj_mask_dict))
                self.visualize_masks()

        if event.key == "enter": # save all cached masks
            if not len(self.obj_mask_dict.keys()) == 0:
                print("[done] number of cached masks:", len(self.obj_mask_dict))
                self.visualize_masks()
            else:
                print("No mask found!")

        if event.key.lower() == "r": # [r]efresh current image
            self.refresh_img_status()
            print("[refreshed]")
            self.ax.cla()
            self.ax.imshow(self.image.astype(np.uint8))
            self.ax.set_title(self.obj_name_list[self.obj_idx])
            plt.show()

if __name__ == "__main__":
    bbs = BBoxSelect()
    img = cv2.cvtColor(cv2.imread("0.png"), cv2.COLOR_BGR2RGB)
    res = bbs.run_gui(img, obj_name_list=["nut", "bolt", "wrench", "stick"])

    SAVE_FOLDER = "./results"
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    for n, m in res.items():
        cv2.imwrite(f"{SAVE_FOLDER}/{n}.png", m)