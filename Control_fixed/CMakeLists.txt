cmake_minimum_required(VERSION 3.22)


# ============================================================
# Project basic settings
# ============================================================

set(CMAKE_PROJECT_NAME Control)


set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)


if (NOT CMAKE_BUILD_TYPE)

    set(
            CMAKE_BUILD_TYPE
            Debug
    )

endif()


# ============================================================
# Toolchain
#
# CLion如果已经通过CMake Profile指定toolchain，
# 就不重复加载。
# ============================================================

if (NOT CMAKE_TOOLCHAIN_FILE)

    include(
            "${CMAKE_SOURCE_DIR}/cmake/gcc-arm-none-eabi.cmake"
    )

endif()


# ============================================================
# Project
# ============================================================

project(
        ${CMAKE_PROJECT_NAME}

        C
        CXX
        ASM
)


set(
        CMAKE_EXPORT_COMPILE_COMMANDS
        TRUE
)


# ============================================================
# Main executable
#
# 所有User_File代码后面统一通过GLOB_RECURSE加入。
#
# 不再单独手动添加：
#
# sorting_storage.cpp
# sorting_storage.h
#
# 这样以后你在User_File下面增加新的.cpp，
# CMake会自动发现。
# ============================================================

add_executable(
        ${CMAKE_PROJECT_NAME}
)


# ============================================================
# STM32CubeMX generated code
#
# CubeMX负责：
#
# Core/Src
# HAL Driver
# startup
# GPIO
# TIM
# UART
# FDCAN
# 等
#
# 不要再次手动GLOB Drivers。
# ============================================================

add_subdirectory(
        cmake/stm32cubemx
)


# ============================================================
# User_File source files
#
# 自动搜索：
#
# User_File/**/*.c
# User_File/**/*.cpp
#
# CONFIGURE_DEPENDS:
#
# 如果以后新增cpp文件，
# CMake会重新扫描。
# ============================================================

file(
        GLOB_RECURSE
        USER_SOURCES

        CONFIGURE_DEPENDS

        "${CMAKE_SOURCE_DIR}/User_File/*.c"
        "${CMAKE_SOURCE_DIR}/User_File/*.cpp"
)


# ============================================================
# 自动排除当前CubeMX没有生成的外设模块
# ============================================================


# ============================================================
# SPI
#
# 当前没有：
#
# Core/Inc/spi.h
#
# 时排除：
#
# SPI
# BMI088
# WS2812
# ============================================================

if (
        NOT EXISTS
        "${CMAKE_SOURCE_DIR}/Core/Inc/spi.h"
)

    message(
            STATUS
            "SPI not enabled -> exclude SPI/BMI088/WS2812 modules"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/1_Middleware/1_Driver/SPI/.*"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/2_Device/BSP/BMI088/.*"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/2_Device/BMI088/.*"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/2_Device/BSP/WS2812/.*"
    )

endif()


# ============================================================
# OCTOSPI
# ============================================================

if (
        NOT EXISTS
        "${CMAKE_SOURCE_DIR}/Core/Inc/octospi.h"
)

    message(
            STATUS
            "OCTOSPI not enabled -> exclude OSPI/W25Q64 modules"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/1_Middleware/1_Driver/OSPI/.*"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/2_Device/BSP/W25Q64JV/.*"
    )

endif()


# ============================================================
# USB CDC
# ============================================================

if (
        NOT EXISTS
        "${CMAKE_SOURCE_DIR}/USB_DEVICE/App/usbd_cdc_if.h"
)

    message(
            STATUS
            "USB CDC not enabled -> exclude USB/VOFA modules"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/1_Middleware/1_Driver/USB/.*"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/2_Device/Vofa/.*"
    )

endif()


# ============================================================
# ADC
# ============================================================

if (
        NOT EXISTS
        "${CMAKE_SOURCE_DIR}/Core/Inc/adc.h"
)

    message(
            STATUS
            "ADC not enabled -> exclude ADC/Power modules"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/1_Middleware/1_Driver/ADC/.*"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/2_Device/BSP/Power/.*"
    )

endif()


# ============================================================
# IWDG
# ============================================================

if (
        NOT EXISTS
        "${CMAKE_SOURCE_DIR}/Core/Inc/iwdg.h"
)

    message(
            STATUS
            "IWDG not enabled -> exclude WDG module"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/1_Middleware/1_Driver/WDG/.*"
    )

endif()


# ============================================================
# FDCAN
#
# 当前Sorting_Storage需要：
#
# 1_Middleware/1_Driver/CAN/drv_can.cpp
#
# 以及：
#
# 2_Device/Motor/Motor_DM/dvc_motor_dm.cpp
#
# 所以即使当前只有FDCAN1，
# 这两个模块也必须保留。
# ============================================================

if (
        NOT EXISTS
        "${CMAKE_SOURCE_DIR}/Core/Inc/fdcan.h"
)

    # ========================================================
    # 连FDCAN都没有启用
    #
    # 此时CAN / DJI / DM全部不能编译。
    # ========================================================

    message(
            STATUS
            "FDCAN not enabled -> exclude CAN/DJI/DM modules"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/1_Middleware/1_Driver/CAN/.*"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/2_Device/Motor/Motor_DJI/.*"
    )


    list(
            FILTER
            USER_SOURCES

            EXCLUDE
            REGEX
            ".*/2_Device/Motor/Motor_DM/.*"
    )

else()

    # ========================================================
    # 已经存在FDCAN
    #
    # 当前工程就是这种情况。
    #
    # CAN driver：
    #       保留
    #
    # Motor_DM：
    #       保留
    #
    # 因为Sorting_Storage需要DM4310。
    # ========================================================

    message(
            STATUS
            "FDCAN enabled -> keep CAN driver and Motor_DM for DM4310"
    )


    # ========================================================
    # 检查FDCAN3
    #
    # 当前没有FDCAN3时，
    # 旧Motor_DJI暂时继续排除。
    #
    # 注意：
    # 不再排除：
    #
    # CAN
    # Motor_DM
    # ========================================================

    file(
            READ

            "${CMAKE_SOURCE_DIR}/Core/Inc/fdcan.h"

            FDCAN_HEADER_CONTENT
    )


    string(
            FIND

            "${FDCAN_HEADER_CONTENT}"

            "hfdcan3"

            FDCAN3_POSITION
    )


    if (
            FDCAN3_POSITION
            EQUAL
            -1
    )

        message(
                STATUS
                "FDCAN3 not enabled -> exclude old DJI motor module only"
        )


        list(
                FILTER
                USER_SOURCES

                EXCLUDE
                REGEX
                ".*/2_Device/Motor/Motor_DJI/.*"
        )

    endif()

endif()


# ============================================================
# Task文件
#
# 当前保留5_Task。
#
# 你现在直接在：
#
# tsk_config_and_callback.cpp
#
# 里面：
#
# robot.Sorting_Storage.Start_Sort(3);
#
# 进行独立测试是可以的。
# ============================================================


# ============================================================
# Add User sources
# ============================================================

target_sources(
        ${CMAKE_PROJECT_NAME}

        PRIVATE

        ${USER_SOURCES}
)


# ============================================================
# Include directories
# ============================================================

target_include_directories(
        ${CMAKE_PROJECT_NAME}

        PRIVATE


        # ----------------------------------------------------
        # User project root
        #
        # 因此可以：
        #
        # #include "3_Robot/sorting_storage.h"
        # ----------------------------------------------------

        "${CMAKE_SOURCE_DIR}/User_File"


        # ----------------------------------------------------
        # CMSIS DSP
        # ----------------------------------------------------

        "${CMAKE_SOURCE_DIR}/Drivers/CMSIS/DSP/Include"
)


# ============================================================
# CMSIS / ARM Math definitions
# ============================================================

target_compile_definitions(
        ${CMAKE_PROJECT_NAME}

        PRIVATE

        ARM_MATH_CM7

        ARM_MATH_MATRIX_CHECK

        ARM_MATH_ROUNDING
)


# ============================================================
# C++ compile options
#
# 当前CMSIS Core版本较旧，
# GCC 13 C++严格检查下：
#
# cmsis_gcc.h
# __copy_table_start__
# __zero_table_start__
#
# 可能产生编译问题。
#
# 这里只对C++添加-fpermissive。
# ============================================================

target_compile_options(
        ${CMAKE_PROJECT_NAME}

        PRIVATE

        $<$<COMPILE_LANGUAGE:CXX>:-fpermissive>
)


# ============================================================
# CMSIS DSP library
# ============================================================

set(
        CMSIS_DSP_LIB

        "${CMAKE_SOURCE_DIR}/Drivers/CMSIS/DSP/Lib/GCC/libarm_cortexM7lfsp_math.a"
)


if (
        EXISTS
        "${CMSIS_DSP_LIB}"
)

    message(
            STATUS

            "CMSIS DSP library found: ${CMSIS_DSP_LIB}"
    )


    target_link_libraries(
            ${CMAKE_PROJECT_NAME}

            "${CMSIS_DSP_LIB}"
    )

else()

    message(
            WARNING

            "CMSIS DSP library not found: ${CMSIS_DSP_LIB}"
    )

endif()


# ============================================================
# Extra linker options
# ============================================================

target_link_options(
        ${CMAKE_PROJECT_NAME}

        PRIVATE

        -Wl,--gc-sections

        -Wl,--print-memory-usage
)


# ============================================================
# Output HEX / BIN
# ============================================================

set(
        HEX_FILE

        "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}.hex"
)


set(
        BIN_FILE

        "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}.bin"
)


# ============================================================
# Build结束后：
#
# elf
# ↓
# hex
# bin
# size
# ============================================================

add_custom_command(
        TARGET
        ${CMAKE_PROJECT_NAME}

        POST_BUILD


        COMMAND
        ${CMAKE_OBJCOPY}

        -Oihex

        $<TARGET_FILE:${CMAKE_PROJECT_NAME}>

        ${HEX_FILE}


        COMMAND
        ${CMAKE_OBJCOPY}

        -Obinary

        $<TARGET_FILE:${CMAKE_PROJECT_NAME}>

        ${BIN_FILE}


        COMMAND
        ${CMAKE_SIZE}

        $<TARGET_FILE:${CMAKE_PROJECT_NAME}>


        COMMENT
        "Generating HEX/BIN files"
)