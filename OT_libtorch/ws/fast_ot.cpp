#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>
#include <pcl/point_cloud.h>

using namespace std;


std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));
    return lidar_data_buffer;
}




/// max_range must be the same as database
void gen_range_image(float * virtual_image, pcl::PointCloud<pcl::PointXYZI>::Ptr current_vertex,
                        float fov_up, float fov_down, int proj_H, int proj_W, int max_range)
{
    int len_arr = proj_W*proj_H;
    fov_up = fov_up * M_PI / 180;
    fov_down = fov_down * M_PI / 180;

    float fov = std::abs(fov_down) + std::abs(fov_up);

    for(int i=0;i<len_arr;i++)
	    virtual_image[i] = -1.0;

    // loop cloud
    for(int p=0; p<current_vertex->points.size(); p++)
    {
        float px = current_vertex->points[p].x;
        float py = current_vertex->points[p].y;
        float pz = current_vertex->points[p].z;
        float depth = sqrt(px*px+py*py+pz*pz);
        if (depth >= max_range || depth <= 0)
          continue;


        float yaw = -std::atan2(py, px);
        float pitch = std::asin(pz / depth);

        float proj_x = 0.5 * (yaw / M_PI + 1.0);
        float proj_y = 1.0 - (pitch + std::abs(fov_down)) / fov;

        proj_x *= proj_W; // in [0.0, W]
        proj_y *= proj_H; // in [0.0, H]

        proj_x = std::floor(proj_x);
        proj_x = std::min(proj_W - 1, static_cast<int>(proj_x));
        proj_x = std::max(0, static_cast<int>(proj_x)); // in [0,W-1]

        proj_y = std::floor(proj_y);
        proj_y = std::min(proj_H - 1, static_cast<int>(proj_y));
        proj_y = std::max(0, static_cast<int>(proj_y)); // in [0,H-1];

        float old_depth = virtual_image[int(proj_y*proj_W + proj_x)];
        if ((depth < old_depth && old_depth > 0) || old_depth < 0)
        {
            virtual_image[int(proj_y*proj_W + proj_x)] = depth;
        }
    }
}




int main()
{

    std::stringstream lidar_data_path;
    lidar_data_path << "../000000.bin";


    std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZI>);

    for (std::size_t i = 0; i < lidar_data.size(); i += 4)
    {
        pcl::PointXYZI point;
        point.x = lidar_data[i];
        point.y = lidar_data[i + 1];
        point.z = lidar_data[i + 2];
        point.intensity = lidar_data[i + 3];
        cloud0->points.push_back(point);
    }

    int width = 900;
    int height = 64;
    float fov_up = 3;
    float fov_down = -25;
    int len_arr = width*height;
    float range_image[len_arr];
    gen_range_image(range_image, cloud0, fov_up, fov_down, height, width, 50);




    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type);
    std::cout<<"cuda support:"<< (torch::cuda::is_available()?"ture":"false")<<std::endl;


    torch::jit::script::Module module = torch::jit::load("../overlapTransformer.pt");
    module.to(torch::kCUDA);
    module.eval();


    torch::Tensor tester  = torch::from_blob(range_image, {1, 1, height, width}, torch::kFloat).to(device);
    double desc_gen_time = 0;
    for (int i=0; i<1000; i++)
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        torch::Tensor result = module.forward({tester}).toTensor();
        result = result.to(torch::kCPU);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        // std::cout << "Processing time = " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000.0 << " sec" <<std::endl;
        desc_gen_time += (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000.0;
    }
    std::cout << "Processing time per frame = " << desc_gen_time/1000.0 << " sec" <<std::endl;
    return 0;
}
