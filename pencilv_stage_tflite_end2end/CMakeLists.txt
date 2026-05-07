cmake_minimum_required(VERSION 3.13.1)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(pencilv_stage_tflite_end2end)

zephyr_compile_options(-fdiagnostics-color=always)
set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

target_include_directories(app PRIVATE
    .
    src
    src/generated
)

# 生成 runner 已切换到 CMSIS-DSP，不再编译 kissfft。
target_sources(app PRIVATE
    src/main.cpp
    src/generated/audio_event_model_runner.cpp
    src/generated/audio_event_model_data.cpp
)
