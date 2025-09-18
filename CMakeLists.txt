cmake_minimum_required(VERSION 3.10)
project(my_project)

# 使用 C++17，libtorch 需要
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找 OpenCV
find_package(OpenCV REQUIRED)

# 找 libtorch（换成你的路径）
set(Torch_DIR "/home/liusi/下载/libtorch-shared-with-deps-2.8.0+cpu/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# 遍历当前目录下的所有 .cpp 文件
file(GLOB SRC_FILES "*.cpp")

# 给每一个 .cpp 文件生成一个同名的可执行文件
foreach(src ${SRC_FILES})
    # 获取文件名（不带路径和后缀）
    get_filename_component(exe ${src} NAME_WE)
    
    # 创建可执行文件
    add_executable(${exe} ${src})

    # 链接 OpenCV 和 libtorch
    target_link_libraries(${exe} PRIVATE ${OpenCV_LIBS} "${TORCH_LIBRARIES}")

    # ABI 设置
    target_compile_definitions(${exe} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)
endforeach()
