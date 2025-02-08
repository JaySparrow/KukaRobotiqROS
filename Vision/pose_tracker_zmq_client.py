import zmq
import numpy as np

class PoseTrackerZMQClient:
    r"""
    Data types of arrays:
        - int: np.int32
        - float: np.float64
        - image: np.uint8
    4 Modes
        1. "INIT": Request the Server to initialize the PoseTracker for each object
            - Send
                * Mode (str) "INIT"
                * Object names (str) e.g. "nut2 bolt2 wrench2_head stick"
                * Camera intrinsics (np.array 3x3) 
            - Receive
                * Names of the objects that have been initialized (str) e.g. "nut2 wrench2_head"
                * bboxCenter_T_localOrigin (np.array NUM_OBJECTSx4x4)
                * bbox (np.array NUM_OBJECTSx2x3)

        2. "ESTIMATE": Segment the (initialized) objects 
                        and request the Server to estimate the pose of each object
            - Do
                * Segment the (initialized) objects
            - Send
                * Mode (str) "ESTIMATE"
                * HxW (np.array 2x1)
                * Image (np.array HxWx3)
                * Depth (np.array HxW)
                * Object Names (str) e.g. "nut2 bolt2 wrench2_head stick"
                * Masks (np.array NUM_OBJECTSxHxW)
            - Receive
                * Pose of each object (np.array NUM_OBJECTSx4x4)

        3. "TRACK": Request the Server to track the pose of each object
            - Send
                * Mode (str) "TRACK"
                * HxW (np.array 2x1)
                * Image (np.array HxWx3)
                * Depth (np.array HxW)
                * Object Names (str) e.g. "nut2 bolt2 wrench2_head stick"
            - Receive
                * Pose of each object (np.array NUM_OBJECTSx4x4)  
        4. "CLOSE": Close the Server
            - Send
                * Mode (str) "CLOSE"
            - Receive
                * msg (str) "closed"  
            - Do
                * Close the connnection
    """
    def __init__(self, server_ip="", port="8080"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://%s:%s" % (server_ip, port))

    def request_init(self, object_names, camera_intrinsics):
        r"""
        Request the Server to initialize the PoseTracker for each object
        """
        mode = "INIT"
        msgs = [
                mode.encode('utf-8'), 
                object_names.encode('utf-8'), 
                camera_intrinsics.astype(np.float64).tobytes()
                ]
        self.socket.send_multipart(msgs)

        msgs_back = self.socket.recv_multipart(flags=0)
        initialized_obj_names_list = msgs_back[0].decode('utf-8').split()
        bboxCenter_T_localOrigin_stack = np.frombuffer(msgs_back[1], dtype=np.float64).reshape(-1, 4, 4) # (num_objects, 4, 4)
        bbox_stack = np.frombuffer(msgs_back[2], dtype=np.float64).reshape(-1, 2, 3) # (num_objects, 2, 3)

        initialized_object_dict = {}
        for obj_name, bboxCenter_T_localOrigin, bbox in zip(initialized_obj_names_list, bboxCenter_T_localOrigin_stack, bbox_stack):
            initialized_object_dict[obj_name] = {"bboxCenter_T_localOrigin": bboxCenter_T_localOrigin, "bbox": bbox}
        return initialized_object_dict
    
    def request_estimate(self, image, depth, object_names_list, masks):
        r"""
        Segment the (initialized) objects and request the Server to estimate the pose of each object
        """
        mode = "ESTIMATE"
        H, W, _ = image.shape
        object_names = " ".join(object_names_list)
        msgs = [
                mode.encode('utf-8'), 
                np.array([H, W], dtype=np.int32).tobytes(), 
                image.flatten().astype(np.uint8).tobytes(), 
                depth.flatten().astype(np.float64).tobytes(), 
                object_names.encode('utf-8'), 
                masks.astype(np.uint8).tobytes()
                ]
        self.socket.send_multipart(msgs)
        msg_back = self.socket.recv()
        return np.frombuffer(msg_back, dtype=np.float64).reshape(-1, 4, 4)

    def request_track(self, image, depth, object_names_list):
        r"""
        Request the Server to track the pose of each object
        """
        mode = "TRACK"
        H, W, _ = image.shape
        object_names = " ".join(object_names_list)
        msgs = [
                mode.encode('utf-8'), 
                np.array([H, W], dtype=np.int32).tobytes(), 
                image.flatten().astype(np.uint8).tobytes(), 
                depth.flatten().astype(np.float64).tobytes(), 
                object_names.encode('utf-8')
                ]
        self.socket.send_multipart(msgs)
        msg_back = self.socket.recv()
        return np.frombuffer(msg_back, dtype=np.float64).reshape(-1, 4, 4)

    def request_close(self):
        r"""
        Close the Server
        """
        mode = "CLOSE"
        self.socket.send(mode.encode('utf-8'), flags=0)
        msg_back = self.socket.recv().decode('utf-8')
        print(f"[Server] {msg_back}")
        self.socket.close()
        print("Connection closed")

    def close(self):
        self.socket.close()
        print("Connection closed")

if __name__ == "__main__":
    from realsense import RealsenseSensor

    from mask_select import MaskSelect
    from vision_utils import visualize_pose_2d

    import cv2

    server_ip = "172.16.71.27"
    port = "8081"
    object_names = "nut2 bolt2 wrench2_head stick"
    extrinsics_file = "./cam_T_base.txt"
    cam_T_base = np.loadtxt(extrinsics_file, dtype=np.float64)

    realsense = RealsenseSensor(device_id="109422062805")
    mask_selecter = MaskSelect()
    pose_tracker_client = PoseTrackerZMQClient(server_ip=server_ip, port=port)

    camera_intrinsics = realsense.cam_K
    initialized_object_dict = pose_tracker_client.request_init(object_names, camera_intrinsics)
    initialized_object_names_str = " ".join(initialized_object_dict.keys())
    if input(f"Objects with existing mesh files on Server: {initialized_object_names_str}. Continue? (y/[n]): ").lower() != "y":
        exit()

    try:
        frame_idx = 0
        while True:
            rgb, depth, is_frame_received = realsense.get_aligned_frames(depth_processed=False)
            if not is_frame_received:
                continue

            if frame_idx == 0:
                masks_dict = mask_selecter.run_gui(rgb, list(initialized_object_dict.keys()))
                object_names = list(masks_dict.keys())
                masks = np.stack(list(masks_dict.values()), axis=0)
                poses = pose_tracker_client.request_estimate(rgb, depth, object_names, masks)
            else:
                poses = pose_tracker_client.request_track(rgb, depth, object_names)

            vis = rgb.copy()
            for obj_name, pose in zip(object_names, poses):
                bboxCenter_T_localOrigin = initialized_object_dict[obj_name]["bboxCenter_T_localOrigin"]
                bbox = initialized_object_dict[obj_name]["bbox"]
                vis = visualize_pose_2d(vis, pose, realsense.cam_K, draw_bbox=True, bbox=bbox, bboxCenter_T_localOrigin=bboxCenter_T_localOrigin)
            cv2.imshow("vis", vis[..., ::-1])
            cv2.waitKey(1)

            frame_idx += 1
    
    except KeyboardInterrupt:
        realsense.stop()
        cv2.destroyAllWindows()
        pose_tracker_client.close()

