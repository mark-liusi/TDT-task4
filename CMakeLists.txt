cmake_minimum_required(VERSION 3.10)
project(my_project)

# 使用 C++17（libtorch 需要）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找 OpenCV
find_package(OpenCV REQUIRED)

# 找 libtorch（保持你的路径；用 Torch_DIR 也可以，用 CMAKE_PREFIX_PATH 亦可）
set(Torch_DIR "/home/liusi/下载/libtorch-shared-with-deps-2.8.0+cpu/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# 找 Eigen3：NO_MODULE 可以优先走官方的 targets（Eigen3::Eigen）
find_package(Eigen3 REQUIRED NO_MODULE)

# 遍历当前目录下的所有 .cpp 文件（CONFIGURE_DEPENDS 让新增/删除源码时自动刷新）
file(GLOB SRC_FILES CONFIGURE_DEPENDS "*.cpp")

# 给每一个 .cpp 文件生成一个同名可执行文件
foreach(src ${SRC_FILES})
    # 获取文件名（不带路径和后缀）
    get_filename_component(exe ${src} NAME_WE)

    # 创建可执行文件
    add_executable(${exe} ${src})

    # 链接 OpenCV、libtorch、Eigen
    # 注意：Eigen 是头文件库，用 Eigen3::Eigen（不要把 include 路径放到 target_link_libraries 里）
    target_link_libraries(${exe} PRIVATE ${OpenCV_LIBS} ${TORCH_LIBRARIES} Eigen3::Eigen)

    # 把 libtorch 推荐的编译标志加进去（里头自带 ABI 宏：_GLIBCXX_USE_CXX11_ABI=?）
    # 用 compile_options 是最省心的方式（其内含的 -D 宏和其他 flags 会被一起传给编译器）
    target_compile_options(${exe} PRIVATE ${TORCH_CXX_FLAGS})

    # 如果你坚持手动指定 ABI 宏，写法应为下面这种（注意不要带 -D）：
    # target_compile_definitions(${exe} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    # 但二选一即可，避免和 TORCH_CXX_FLAGS 里的设置冲突。
endforeach()
# wo shi 