import pyrealsense2 as rs
import numpy as np
import cv2

class RealsenseSensor:
    def __init__(self, device_id: str, num_jump_frames: int=100, h: int=480, w: int=640):
        # Create a pipeline
        pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream different resolutions of color and depth streams
        config = rs.config()
        config.enable_device(device_id)
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        # Make sure that the device contains a color sensor
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                found_rgb = True
                break
        if not found_rgb:
            print(f"The demo requires Depth camera with Color sensor {device_id}")
            exit(0)
        # Resolution
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = pipeline.start(config)
        print("pipeline started")

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # Get the intrinsics of color
        profile = pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        cam_K = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                           [0, color_intrinsics.fy, color_intrinsics.ppy],
                           [0, 0, 1]])
        
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        # variables
        self.pipeline = pipeline
        self.align = align
        self.cam_K = cam_K
        self.depth_scale = depth_scale
        self.device_id = device_id
        self.h = h
        self.w = w
        
        # Jump frames
        num_frames = 0
        while num_frames < num_jump_frames:
            _, _, is_frame_received = self.get_aligned_frames()
            if is_frame_received:
                num_frames += 1

    def get_aligned_frames(self, depth_processed: bool=True):
        try: 
            # Get frameset of color and depth
            frames = self.pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                return None, None, False
            
            # Convert images to numpy arrays
            rgb = np.asarray(color_frame.get_data(), dtype=np.uint8)[..., ::-1]
            depth = np.asarray(aligned_depth_frame.get_data())

            # Resize frames
            if self.w != 640 or self.h != 480:
                rgb = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

            if depth_processed:
                depth = depth * self.depth_scale
                depth[(depth<0.001) | (depth>=np.inf)] = 0

            return rgb, depth, True
        except Exception as e:
            self.stop()
            print("unable to get frames:", repr(e))
            return None, None, False

    def stop(self):
        self.pipeline.stop()

if __name__ == "__main__":
    import cv2, os

    root = "./cluster2"
    rgb_root = os.path.join(root, "rgb")
    depth_root = os.path.join(root, "depth")
    if not os.path.isdir(rgb_root):
        os.makedirs(rgb_root)
    if not os.path.isdir(depth_root):
        os.makedirs(depth_root)
    
    realsense = RealsenseSensor(device_id="109422062805")
    
    try:
        np.savetxt(os.path.join(root, "cam_K.txt"), realsense.cam_K)
        i = 0
        n = 0
        while True:
            rgb, depth, is_frame_received = realsense.get_aligned_frames()
            if i % 10 == 0:
                if is_frame_received:
                    cv2.imwrite(os.path.join(rgb_root, f"{i}.png"), rgb[:, :, ::-1])
                    cv2.imwrite(os.path.join(depth_root, f"{i}.png"), depth)
                    n += 1
                else:
                    print(f"[{i}] No frame received")
                print(f"saved {n} images")
            i += 1
                
    finally:
       realsense.stop()