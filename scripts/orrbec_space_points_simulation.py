#! /home/zjy/anaconda3/envs/yolo3D_py38/bin/python
# coding=utf-8


import rospy
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge, CvBridgeError
import threading

# 全局变量
bridge = CvBridge()
rgb_image = None
depth_image = None
image_lock = threading.Lock()
depth_scale = 0.001  # 深度缩放因子（mm→m），默认1mm=0.001m

class Space_posture:
    def __init__(self):      
        # Model configuration
        self.model_config = {
            'model_path': r"/home/zjy/roboncon2025_ws/src/yolo_depth_pkg/scripts/best.pt"
        }
        self.predict_config = {
            'img_size': 640,
            'conf': 0.5,
            'iou': 0.5
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(self.model_config['model_path']).to(self.device)
        
        rospy.loginfo("YOLO Model Loaded Successfully")
        
        # ROS Node & Publisher/Subscriber
        rospy.init_node("space_posture_node", anonymous=True)
        self.rate = rospy.Rate(15)
        self.puber = rospy.Publisher("/orbbec_kfs_node", PointCloud2, queue_size=10)
        
        # 订阅RGB和Depth话题
        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback, queue_size=1)
        
        rospy.loginfo("Space_posture Node Initialized (订阅模式)")

        # 相机内参（从camera_info提取的精准值）
        self.fx = 386.6107
        self.fy = 386.0946
        self.cx = 320.9239
        self.cy = 243.7342
        
        # 深度阈值
        self.max_depth = 6.0  # 最大有效深度6m


    def rgb_callback(self, msg):
        """RGB图像回调函数"""
        global rgb_image
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            with image_lock:
                rgb_image = cv_image
        except CvBridgeError as e:
            rospy.logerr(f"RGB图像转换失败: {e}")

    def depth_callback(self, msg):
        """深度图像回调函数"""
        global depth_image
        try:
            # 深度图像默认是16位无符号整数（mm单位）
            cv_image = bridge.imgmsg_to_cv2(msg, "16UC1")
            with image_lock:
                depth_image = cv_image
        except CvBridgeError as e:
            rospy.logerr(f"深度图像转换失败: {e}")

    def masks2mask_list(self, results):  # 假设这是类中的方法，保留self；如果是独立函数，删除self
        """
        将YOLO-seg的Results对象转换为掩码列表（包含每个物体的掩码+类别ID）
        :param results: YOLO推理返回的Results对象/列表
        :return: mask_list - 每个元素是字典{'mask': 掩码矩阵, 'cls': 类别ID}
        """
        for result in results:
            # 跳过空结果/无掩码/无检测框的情况
            if (result is None  
                or result.masks is None or len(result.masks.data) == 0
                or result.boxes is None or len(result.boxes.cls) == 0):
                rospy.logwarn("No valid masks/boxes in current result")
                continue

            # 获取原图尺寸，用于掩码缩放
            orig_h, orig_w = result.orig_shape
            # 遍历每个物体的掩码（核心：用len获取物体数量）
            for i in range(len(result.masks.data)):
                mask_list = []
                try:
                    # 1. 提取并处理掩码：CPU + numpy + 二值化 + 缩放到原图尺寸
                    mask = result.masks.data[i].cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8)
                    # mask = np.resize(mask, (orig_h, orig_w))  # 缩放到原图尺寸

                    # 2. 提取对应类别ID（确保i不越界）
                    if i >= len(result.boxes.cls):
                        rospy.logwarn(f"Index {i} out of range for boxes.cls, skip")
                        continue
                    cls_id = int(result.boxes.cls[i])

                    # 3. 添加到掩码列表
                    mask_list.append({
                        'mask': mask,
                        'cls': cls_id
                    })
                    # rospy.loginfo(f"Add mask for cls {cls_id}, mask shape: {mask.shape}")
                    return mask_list
                except Exception as e:
                    rospy.logerr(f"Failed to process mask {i}: {str(e)}", exc_info=True)
                    continue  # 单个物体处理失败，跳过继续处理下一个

        
            
    
    def mask_list_filter(self, mask_list, depth_image):

        hwd_points = []
        mask_cls_list = []
        if depth_image is None or depth_image.size == 0:
            rospy.logerr("depth image is empty")
            return None
        for mask_info in mask_list:
            mask = mask_info['mask']
            mask = mask[:400,:]
            mask_area = cv2.countNonZero(mask) if mask.dtype == np.uint8 else np.count_nonzero(mask)
            if mask_area < 50:
                rospy.loginfo("mask area too low")
                continue
            
            ys , xs = np.where(mask>0)
            
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            center_d = depth_image[cy,cx]
            if center_d <= 0 or center_d > 6000:
                rospy.logwarn("center depth is error")
                continue
            depth_THRESH = 400
            
            """向量化"""
            depth_mm = depth_image[ys,xs]
            valid_depth_idx = (depth_mm > 0) & (depth_mm < 6000) & (np.abs(depth_mm - center_d) < depth_THRESH)
            
            valid_x = xs[valid_depth_idx]
            valid_y = ys[valid_depth_idx]
            valid_z = depth_mm[valid_depth_idx]*depth_scale
            
            if len(valid_z) < 200:
                rospy.loginfo("点数太小")
                continue
            else:
                hwd_points.append({
                    'points':np.stack([valid_x,valid_y,valid_z],axis=1).astype(np.float16),
                    'cls':mask_info['cls']
                })
                mask_cls_list.append({
                    'cx':cx,
                    'cls':mask_info['cls']
                })
        
        mask_cls_list = sorted(mask_cls_list,key=lambda x:x['cx'])
        mask_cls_list = [item['cls'] for item in mask_cls_list]
        return (hwd_points , mask_cls_list)

    def points_list2Tensor(self,hwd_points):
        if not hwd_points:
            rospy.loginfo("no valid points data")
            return None
        
        array_xyd_list = []
        for p in hwd_points:
            array_xyd_list.append(p['points'])
        array_xyd = np.vstack(array_xyd_list).astype(np.float16)
        tensor_xyd = torch.from_numpy(array_xyd).to(self.device)
        rospy.loginfo(f"提取到有效mask点数：{tensor_xyd.shape[0]}")
        return tensor_xyd

    def trans2Space(self, tensor_xyd):
        """
        Pixel_coord To Space 3_dims coord（像素坐标→空间3D坐标）
        :param tensor_xyd: (N,3) tensor [x,y,depth_m]
        :return: (N,3) tensor [X,Y,Z]（空间坐标，m）
        """
        if tensor_xyd is None or tensor_xyd.shape[0] == 0:
            rospy.logwarn("tensor_xyd为空，跳过3D转换")
            return None
        
        N = tensor_xyd.shape[0]
        device = tensor_xyd.device
        
        # 1. 提取像素坐标和深度值
        x = tensor_xyd[:, 0]  # (N,) 像素x
        y = tensor_xyd[:, 1]  # (N,) 像素y
        z = tensor_xyd[:, 2]  # (N,) 深度值（m）
        
        # # 2. 构造像素偏移矩阵 (x-cx, y-cy) → (N,2,1)
        # pixel_offset = torch.stack([x - self.cx, y - self.cy], dim=1).unsqueeze(-1)
        
        # # 3. 构造焦距倒数对角矩阵 (N,2,2)
        # inv_focal = torch.diag(torch.tensor([1/self.fx, 1/self.fy], device=device))  # (2,2)
        # inv_focal_batch = inv_focal.unsqueeze(0).repeat(N, 1, 1)  # (N,2,2)
        
        # # 4. 构造深度对角矩阵 (N,2,2)
        # depth_matrix = torch.diag_embed(torch.stack([z, z], dim=1))  # (N,2,2)
        
        # # 5. 批量矩阵运算计算X,Y
        # xy_3d = torch.bmm(inv_focal_batch, torch.bmm(depth_matrix, pixel_offset)).squeeze(-1)
        
        # # 6. 拼接Z轴得到3D坐标
        # Z = z.unsqueeze(1)  # (N,1)
        # tensor_xyz = torch.cat([xy_3d, Z], dim=1)  # (N,3) [X,Y,Z]
        # 广播计算X/Y，替代批量矩阵
        X = (x - self.cx) * z / self.fx
        Y = (y - self.cy) * z / self.fy
        
        tensor_xyz = torch.stack([X, Y, z], dim=1)  # (N,3)
        return tensor_xyz


    def _publish_pointcloud(self, tensor_xyz):
        """
        发布3D点云到ROS话题
        :param tensor_xyz: (N,3) tensor [X,Y,Z]
        """
        if tensor_xyz is None or tensor_xyz.shape[0] == 0:
            rospy.logwarn("无有效3D点云，跳过发布")
            return
        
        # 转换为numpy数组（CPU）
        if tensor_xyz.is_cuda:
            points_3d_np = tensor_xyz.cpu().numpy()
        else:
            points_3d_np = tensor_xyz.numpy()
        
        # 定义点云字段（x/y/z，float32）
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # 构造ROS头信息
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_link"  # 匹配相机坐标系
        
        # 创建并发布点云消息
        points_msg = point_cloud2.create_cloud(header, fields, points_3d_np)
        self.puber.publish(points_msg)
        rospy.loginfo(f"发布点云成功，点数：{points_3d_np.shape[0]}")

    def run(self):
        """主循环：订阅图像→分割→3D转换→发布点云"""
        rospy.loginfo("进入主循环，等待订阅RGB/Depth图像...")
        
        while not rospy.is_shutdown():
            try:
                # 检查是否获取到图像
                with image_lock:
                    current_rgb = rgb_image.copy() if rgb_image is not None else None
                    current_depth = depth_image.copy() if depth_image is not None else None
                
                if current_rgb is None or current_depth is None:
                    rospy.loginfo("等待RGB/Depth图像数据...")
                    self.rate.sleep()
                    continue


                # 1. YOLO分割推理
                results = self.model.predict(
                    source=current_rgb,
                    imgsz=self.predict_config['img_size'],
                    conf=self.predict_config['conf'],
                    iou=self.predict_config['iou'],
                    verbose=False  # 关闭冗余输出
                )

                # 2. 提取mask列表
                mask_list = self.masks2mask_list(results)
                if not mask_list:
                    rospy.logwarn("未检测到目标mask，跳过")
                    self.rate.sleep()
                    continue

                # 3. mask过滤和点提取
                filter_result = self.mask_list_filter(mask_list, current_depth)
                if filter_result is None:
                    self.rate.sleep()
                    continue
                hwd_points , mask_cls_list = filter_result

                # 4. mask转xyd tensor
                tensor_xyd = self.points_list2Tensor(hwd_points)

                # 5. 像素坐标转3D空间坐标
                tensor_xyz = self.trans2Space(tensor_xyd)

                # 6. 发布点云
                self._publish_pointcloud(tensor_xyz)

                # 7. 控制循环频率
                self.rate.sleep()

            except Exception as e:
                rospy.logerr(f"主循环异常：{str(e)}", exc_info=True)
                self.rate.sleep()
                continue

        rospy.loginfo("节点退出")

if __name__ == "__main__":
    try:
        # 启动节点
        node = Space_posture()
        node.run()

    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断，退出")
    except Exception as e:
        rospy.logerr(f"节点启动失败：{str(e)}", exc_info=True)
    finally:
        rospy.loginfo("程序正常退出")
