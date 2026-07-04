cmake_minimum_required(VERSION 3.10)
project(proc_ransac_c)

# OpenCVを探す
find_package(OpenCV REQUIRED)

# OpenCVのインクルードパスを表示（確認用）
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    libraries: ${OpenCV_LIBS}")

# デバッグビルド用オプション
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")

# リリースビルド用オプション
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

# includeパス追加
include_directories(${OpenCV_INCLUDE_DIRS}
                    include)

# 実行ファイルの追加
add_executable(proc_ransac_c.elf 
               src/main.cpp
            #    src/main2.cpp
               src/fig.cpp
               src/debug.cpp)

# OpenCVのライブラリをリンク
target_link_libraries(proc_ransac_c.elf 
                      ${OpenCV_LIBS})
