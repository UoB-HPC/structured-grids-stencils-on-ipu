set(CMAKE_CXX_STANDARD 17)


include_directories(lib/lodepng)
include_directories(lib/cxxopts)

set(Main_SRCS
        include/GraphcoreUtils.hpp
        include/LbmParams.hpp
        include/LatticeBoltzmannUtils.hpp
        include/StructuredGridUtils.hpp
        include/ImageUtils.hpp
        )

add_executable(lbm_low_level_aos
        LbmAoS.cpp
        ${Main_SRCS}
        )

add_executable(gb_cpu
        GaussianBlurCpu.cpp
        lib/lodepng/lodepng.cpp
        ${Main_SRCS}
        )

add_executable(gb_poplibs
        GaussianBlurPoplibs.cpp
        lib/lodepng/lodepng.cpp
        ${Main_SRCS}
        )

add_executable(gb_low_level
        GaussianBlurLowLevel.cpp
        lib/lodepng/lodepng.cpp
        ${Main_SRCS}
        )

add_executable(halo_regions
        HaloRegionApproaches.cpp
        ${Main_SRCS}
        )

add_executable(viz_tile_map
        VisualiseTileMapping.cpp
        lib/lodepng/lodepng.cpp
        ${Main_SRCS}
        )

add_executable(tile_mapping_stats
        TileMappingStats.cpp
        ${Main_SRCS}
        )


target_link_libraries(lbm_low_level_aos
        poplar
        poputil
        poplin
        )

target_link_libraries(gb_cpu
        PUBLIC OpenMP::OpenMP_CXX
        )

target_link_libraries(gb_poplibs
        poplar
        poputil
        poplin
        popfloat
        )

target_link_libraries(gb_low_level
        poplar
        poputil
        popfloat
        )

target_link_libraries(halo_regions
        poplar
        popops
        poputil
        )

configure_file(codelets/D2Q9Codelets.cpp codelets/D2Q9Codelets.cpp COPYONLY)
configure_file(codelets/GaussianBlurCodelets.cpp codelets/GaussianBlurCodelets.cpp COPYONLY)
configure_file(codelets/HaloRegionApproachesCodelets.cpp codelets/HaloRegionApproachesCodelets.cpp COPYONLY)
configure_file(codelets/GaussianBlurCodeletsVectorised.cpp codelets/GaussianBlurCodeletsVectorised.cpp COPYONLY)
