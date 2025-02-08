import os
import cv2
import numpy as np
from ultralytics import SAM
import matplotlib.pyplot as plt
import shutil

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image.astype(np.uint8))
    
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size)  

def get_file_index(filename):
    return int(filename[:filename.index(".png")])

class MaskSelect:
    def __init__(self, sam_checkpoint):
        self.predictor = SAM(sam_checkpoint)

        self.fig = None

    def run_gui(self, img, obj_name_list=[]):
        self.fig, self.ax = plt.subplots()

        self.click_call = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.key_call = self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

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
        self.labels = []
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
            self.fig.canvas.mpl_disconnect(self.click_call)
            self.fig.canvas.mpl_disconnect(self.key_call)
            plt.close()
    
    def predict(self, img, points, labels):
        print(f"[predict] points={points} | labels={labels}")
        H, W, _ = img.shape
        res = self.predictor(img, points=[points], labels=[labels])[0]
        mask = np.zeros((H, W), np.uint8)
        for ri, r in enumerate(res):
            c = r[0]
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        return mask # (h, w)
    
    def visualize_masks(self):
        self.ax.cla()
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        for obj_mask in self.obj_mask_dict.values():
            mask += obj_mask
        self.ax.imshow(mask)
        plt.show(block=True)

    def on_click(self, event):
        if not event.dblclick:
            if event.button == 1: # left click (foreground)
                if not isinstance(event.xdata, float) and not isinstance(event.ydata, float):
                    print(f"({event.xdata}, {event.ydata}) invalid click!")
                    return
                self.points.append([event.xdata, event.ydata])
                self.labels.append(1)
            if event.button == 3: # right click (background)
                if not isinstance(event.xdata, float) and not isinstance(event.ydata, float):
                    print(f"({event.xdata}, {event.ydata}) invalid click!")
                    return
                self.points.append([event.xdata, event.ydata])
                self.labels.append(0)

            points = np.array(self.points)
            labels = np.array(self.labels)
            self.current_mask = self.predict(self.image, self.points, self.labels)

            self.ax.imshow(self.image.astype(np.uint8))
            show_points(points, labels, self.ax)
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
                    # self.close()

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
                # self.close()

        if event.key == "enter": # save all cached masks
            if not len(self.obj_mask_dict.keys()) == 0:
                print("[done] number of cached masks:", len(self.obj_mask_dict))
                self.visualize_masks()
                # self.close()
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
    sam_checkpoint = "/home/robotics/kuka_workspace/Kuka_Basics/melodic/src/kuka_utils/tutorial/nodes/KukaRobotiqROS/Vision/sam2_b.pt"
    ms = MaskSelect(sam_checkpoint)
    img = cv2.cvtColor(cv2.imread("/home/robotics/yuhan/Tools/test_record/rgb/885.png"), cv2.COLOR_BGR2RGB)
    res = ms.run_gui(img, obj_name_list=["nut", "bolt", "wrench", "stick"])

    SAVE_FOLDER = "./results"
    shutil.rmtree(SAVE_FOLDER)
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    for n, m in res.items():
        cv2.imwrite(f"{SAVE_FOLDER}/{n}.png", m)