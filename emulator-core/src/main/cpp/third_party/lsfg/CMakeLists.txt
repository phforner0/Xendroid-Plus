add_library(xendroid-lsfg STATIC
    lsfg_alpha.cpp lsfg_beta.cpp lsfg_gamma.cpp lsfg_delta.cpp
    lsfg_common.cpp lsfg_mipmaps.cpp lsfg_generate.cpp lsfg_shaders.cpp
    lsfg_chain.cpp lsfg_engine.cpp lsfg_pacer.cpp lsfg_governor.cpp
    lsfg_dll.cpp lsfg_dxbc.cpp lsfg_vkd.cpp)
target_include_directories(xendroid-lsfg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xendroid-lsfg PRIVATE lsfg-dxbc log dl)
target_include_directories(xendroid-lsfg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../xenia/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../xenia/third_party/Vulkan-Headers/include)
target_compile_features(xendroid-lsfg PRIVATE cxx_std_20)
