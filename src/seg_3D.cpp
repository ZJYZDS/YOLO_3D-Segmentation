#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <sensor_msgs/PointCloud2.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing.h>

#include <Eigen/Dense>
#include <pcl/common/transforms.h>
using namespace message_filters;
using namespace sensor_msgs;

// 统一使用带RGB的点类型，兼容原XYZ操作
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointCloud<PointRGB> PointCloudRGB;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

ros::Publisher pub;
ros::Publisher debug_pub;

const float vox_size_c = 0.05f;
const int MEANK = 50;
const float STDDEV_THRESH = 1.2f;

//
const float plane_dist_thresh = 0.05f;
const float z_min_thresh = -0.4f;
const float z_max_thresh = 1.2f;
const float vox_size_l = 0.1f;

// loacl clustering
const float seed_nearest_thresh = 0.1f;
const float search_r = 0.15f;
const float expand_dist_thresh = 0.05f;
const int min_neighbor_num = 15;
const int min_cluster_size = 20;
// 预处理相机点云（保持原逻辑，处理XYZ点云）
void preprocess_camera(PointCloudT::Ptr& cloud)
{
    static pcl::VoxelGrid<PointT> vox;
    vox.setInputCloud(cloud);
    vox.setLeafSize(vox_size_c,vox_size_c,vox_size_c);
    vox.filter(*cloud);

    static pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(MEANK);
    sor.setStddevMulThresh(STDDEV_THRESH);
    sor.filter(*cloud);
}

// Z轴阈值过滤（适配XYZ点云，保持原逻辑）
void filterByZThreshold(PointCloudT::Ptr& in_cloud,PointCloudT::Ptr& out_cloud)
{
    for(int i=0 ; i < in_cloud->size(); ++i )
    {
        double x = in_cloud->at(i).x;
        double y = in_cloud->at(i).y;
        double z = in_cloud->at(i).z;
        
        if( z > z_max_thresh || z < z_min_thresh )
        {
            out_cloud->push_back(in_cloud->points[i]);
            in_cloud->erase(in_cloud->begin() + i);      
        }
    }
}

// RANSAC平面过滤（注释保留，逻辑未改）
void RANSAC_filter(PointCloudT::Ptr& cloud_out)
{
    if(cloud_out->empty()) return ;
    // 原RANSAC逻辑保留，暂未启用
}

// 预处理激光雷达点云（处理XYZ点云，之后转换为RGB点云）
void preprocess_lidar(PointCloudT::Ptr& cloud)
{
    PointCloudT::Ptr cloud_z_filtered (new PointCloudT);
    filterByZThreshold(cloud,cloud_z_filtered);
    static pcl::VoxelGrid<PointT> vox;
    vox.setInputCloud(cloud);
    vox.setLeafSize(vox_size_l,vox_size_l,vox_size_l);
    vox.filter(*cloud);
}

// 坐标变换（处理XYZ点云，保持原逻辑）
PointCloudT::Ptr tf_(const PointCloudT::ConstPtr& cloud)
{
    if(!cloud || cloud->empty())
    {
        ROS_WARN("[tf_] Input pointcloud is empty");
        return PointCloudT::Ptr(new PointCloudT);
    }
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    Eigen::Matrix3f R;
    R<< 0, 0, 1,
        -1, 0, 0,
        0, -1,0;
    T.linear() = R;
    T.translation()<< 0.0438f, 0.0f, -0.0635f;
    Eigen::Matrix4f T_matrix = T.matrix();
    
    PointCloudT::Ptr cloud_tf (new PointCloudT);
    pcl::transformPointCloud(*cloud,*cloud_tf,T_matrix);

    if(cloud_tf->empty())
    {
        ROS_INFO("pointcloud_tf is empty");
        return nullptr;
    }
    return cloud_tf;
}

// 计算欧式距离（适配XYZ点）
inline float calcluateEdist(const PointT& p1 , const PointT& p2)
{
    return sqrt(std::pow(p1.x-p2.x,2) + std::pow(p1.y-p2.y,2) + std::pow(p1.z-p2.z,2));
}


/**
 * 修改后的局部聚类函数
 * @param filtered_l 激光雷达点云（RGB类型，输出：所有点保留，特征点标注RGB）
 * @param seed_cloud 种子点云（相机预处理后的点云）
 */
void local_clustering(PointCloudRGB::Ptr& filtered_l, 
                    PointCloudT::Ptr& seed_cloud)
{
    if (filtered_l == nullptr || filtered_l->empty()) {
        PCL_ERROR("[local_clustering] 激光雷达点云为空！\n");
        return;
    }
    if (seed_cloud == nullptr || seed_cloud->empty()) {
        PCL_WARN("[local_clustering] 种子点云为空！\n");
        return;
    }

    // 初始化所有点为黑色
    for (auto& p : filtered_l->points) {
        p.r = static_cast<unsigned char>(255);
        p.g = static_cast<unsigned char>(255);
        p.b = static_cast<unsigned char>(255);
    }

    // 构建KD树
    PointCloudT::Ptr lidar_xyz(new PointCloudT);
    pcl::copyPointCloud(*filtered_l, *lidar_xyz);
    pcl::KdTreeFLANN<PointT>::Ptr kdtree ( new pcl::KdTreeFLANN<PointT>() );
    kdtree->setInputCloud(lidar_xyz);

    std::vector<bool> is_processed(filtered_l->size(), false);

    // 遍历种子点
    for(const auto& seed_p : seed_cloud->points)
    {
        std::vector<int> seed_nearest_idx(1);
        std::vector<float> seed_nearest_dist(1);
        int search_ret = kdtree->nearestKSearch(seed_p, 1, seed_nearest_idx, seed_nearest_dist);

        if(search_ret == 0 || seed_nearest_dist[0] > seed_nearest_thresh){
            continue;
        }

        int start_idx = seed_nearest_idx[0];
        if(is_processed[start_idx]) continue;

        std::queue<int> expand_queue;
        expand_queue.push(start_idx);
        is_processed[start_idx] = true;
        int current_cluster_size = 1;

        // 第一次遍历：统计聚类大小
        while(!expand_queue.empty())
        {
            int curr_idx = expand_queue.front();
            expand_queue.pop();
            const PointT& curr_point = lidar_xyz->points[curr_idx];

            std::vector<int> neighbor_indices;
            std::vector<float> neighbor_dist;
            kdtree->radiusSearch(curr_point, search_r, neighbor_indices, neighbor_dist);

            if(neighbor_indices.size() < min_neighbor_num) continue;

            for (size_t i = 0; i < neighbor_indices.size(); i++)
            {
                int neighbor_idx = neighbor_indices[i];
                if(is_processed[neighbor_idx]) continue;
                const PointT& neighbor_point = lidar_xyz->points[neighbor_idx];
                float dist = calcluateEdist(curr_point, neighbor_point);
                if(dist > expand_dist_thresh) continue;

                is_processed[neighbor_idx] = true;
                current_cluster_size++;
                expand_queue.push(neighbor_idx);
            }
        }

        // 聚类大小达标，标注RGB
        if (current_cluster_size >= min_cluster_size) {
            std::queue<int> re_expand_queue;
            re_expand_queue.push(start_idx);
            std::vector<bool> temp_processed(filtered_l->size(), false);
            temp_processed[start_idx] = true;

            while(!re_expand_queue.empty()) {
                int curr_idx = re_expand_queue.front();
                re_expand_queue.pop();
                // 标注为red色
                filtered_l->points[curr_idx].r = 0;
                filtered_l->points[curr_idx].g = 0;
                filtered_l->points[curr_idx].b = 0;

                std::vector<int> neighbor_indices;
                std::vector<float> neighbor_dist;
                kdtree->radiusSearch(lidar_xyz->points[curr_idx], search_r, neighbor_indices, neighbor_dist);

                for (size_t i = 0; i < neighbor_indices.size(); i++) {
                    int neighbor_idx = neighbor_indices[i];
                    if(!is_processed[neighbor_idx] || temp_processed[neighbor_idx]) continue;
                    temp_processed[neighbor_idx] = true;
                    re_expand_queue.push(neighbor_idx);
                }
            }
            ROS_INFO("Valid cluster: %d points, marked as white", current_cluster_size);
        } else {
            // 重置小聚类标记
            std::queue<int> reset_queue;
            reset_queue.push(start_idx);
            while(!reset_queue.empty()) {
                int curr_idx = reset_queue.front();
                reset_queue.pop();
                is_processed[curr_idx] = false;

                std::vector<int> neighbor_indices;
                std::vector<float> neighbor_dist;
                kdtree->radiusSearch(lidar_xyz->points[curr_idx], search_r, neighbor_indices, neighbor_dist);

                for (size_t i = 0; i < neighbor_indices.size(); i++) {
                    int neighbor_idx = neighbor_indices[i];
                    if(is_processed[neighbor_idx]) {
                        is_processed[neighbor_idx] = false;
                        reset_queue.push(neighbor_idx);
                    }
                }
            }
        }
    }
}

void callback_sync(const PointCloud2ConstPtr& camera_msg , const PointCloud2ConstPtr& lidar_msg)
{
    if(camera_msg->data.empty() || lidar_msg->data.empty())
    {
        ROS_WARN("msg exist empty data");
        return ;
    }

    ROS_INFO("time diff is : %.4f",fabs((camera_msg->header.stamp - lidar_msg->header.stamp).toSec()));
    
    // 1. 解析相机点云（XYZ类型）
    PointCloudT::Ptr camera_pc (new PointCloudT);
    pcl::fromROSMsg(*camera_msg , *camera_pc);

    // 2. 解析激光雷达点云（先XYZ，再转换为RGB类型）
    PointCloudT::Ptr lidar_pc_xyz (new PointCloudT);
    pcl::fromROSMsg(*lidar_msg , *lidar_pc_xyz);

    // 3. 预处理
    preprocess_camera(camera_pc);
    preprocess_lidar(lidar_pc_xyz);

    // 4. 相机点云坐标变换
    sensor_msgs::PointCloud2 temp_debug_msg;
    
    
    PointCloudT::Ptr seed_cloud = tf_(camera_pc);
    pcl::toROSMsg(*seed_cloud , temp_debug_msg);
    temp_debug_msg.header.stamp = ros::Time::now();
    temp_debug_msg.header.frame_id = "frame_link";

    // 5. 将激光XYZ点云转换为RGB点云（用于标注）
    PointCloudRGB::Ptr lidar_pc_rgb (new PointCloudRGB);
    pcl::copyPointCloud(*lidar_pc_xyz, *lidar_pc_rgb);

    // 6. 局部聚类：标注特征点RGB，保留所有激光点
    local_clustering(lidar_pc_rgb, seed_cloud);

    // 7. 发布标注后的激光点云
    sensor_msgs::PointCloud2 out_msg;
    pcl::toROSMsg(*lidar_pc_rgb, out_msg);
    out_msg.header.stamp = ros::Time::now();
    out_msg.header.frame_id = "livox_frame";

    pub.publish(out_msg);
    debug_pub.publish(temp_debug_msg);
    ROS_INFO("pub success, total points: %zu, feature points will be white", lidar_pc_rgb->size());
}

// 未启用的扩散聚类（保留原注释）
const float ds_r = 0.03f;
const int min_dc_num = 50;
// void diffuse_clustering(PointCloudT::Ptr& lidar_pc,
//                         PointCloudT::Ptr& seed_pc,
//                         PointCloudRGB::Ptr& out_pc)
//     {
//         // 原逻辑保留
//     }

int main(int argc , char** argv)
{
    ros::init(argc,argv,"seg_3d_node");
    ros::NodeHandle nh;
    Subscriber<PointCloud2> camera_sub(nh,"/orbbec_kfs_node",4);
    Subscriber<PointCloud2> lidar_sub(nh,"/livox/lidar",4);
    
    typedef sync_policies::ApproximateTime<PointCloud2 , PointCloud2> Sync;
    Synchronizer<Sync> sync(Sync(10),camera_sub, lidar_sub);
    sync.setMaxIntervalDuration(ros::Duration(0.1));
    ROS_INFO("we get sync");
    sync.registerCallback(boost::bind(&callback_sync,_1,_2));

    pub = nh.advertise<sensor_msgs::PointCloud2>("/seg_3d",10);
    debug_pub = nh.advertise<sensor_msgs::PointCloud2>("/debug",10);
    ros::spin();
}
