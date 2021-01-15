//
// Created by Thorben Louw on 23/06/2020.
//


#ifndef LBM_GRAPHCORE_DOUBLEROLL_H

#define LBM_GRAPHCORE_DOUBLEROLL_H


#include <cstdlib>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <iomanip>
#include <iostream>
#include <poplar/Program.hpp>
#include <popops/Reduce.hpp>
#include <fstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <poputil/Broadcast.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace popops;

typedef std::tuple<size_t, size_t> SliceRange;
typedef std::tuple<SliceRange, SliceRange> SliceRegion;

typedef struct {
    std::vector<SliceRegion> src;
    std::vector<SliceRegion> dst;
} RollSliceMapping;

auto toDstSlices(const std::tuple<size_t, size_t> &dims,
                 const std::tuple<int, int> &roll_offsets, const SliceRegion &sp) -> SliceRegion {
    const auto[roll_x, roll_y] = roll_offsets;
    const auto[lx, ly] = dims;
    auto const &[xSlice, ySlice] = sp;
    auto const &[xFrom, xTo] = xSlice;
    auto const &[yFrom, yTo] = ySlice;
    auto dst_xFrom = (lx + xFrom - roll_x) % lx;
    auto dst_xTo = (lx + xTo - roll_x) % lx;
    dst_xTo = dst_xTo == 0 ? lx : dst_xTo;
    auto dst_yFrom = (ly + yFrom - roll_y) % ly;
    auto dst_yTo = (ly + yTo - roll_y) % ly;
    dst_yTo = dst_yTo == 0 ? ly : dst_yTo;

    return {
            {dst_xFrom, dst_xTo},
            {dst_yFrom, dst_yTo}};
}

auto determineSrcAndDstSlices(const std::tuple<size_t, size_t> &dims,
                              const std::tuple<int, int> &roll_offsets) -> RollSliceMapping {
    std::cerr << __FUNCTION__ << std::endl;
    const auto[roll_x, roll_y] = roll_offsets;
    const auto[lx, ly] = dims;

    assert(roll_x > -2 && roll_x < 2);
    assert(roll_y > -2 && roll_y < 2);

    auto s = Sequence();


//    std::cerr << "lx " << lx << "ly " << ly << " " << "rollx " << roll_x << "roll_y " << roll_y << " " << __FUNCTION__
//              << std::endl;


    auto src_slices = std::vector<SliceRegion>{
            // x slice, y slice
            {{1,      lx - 1}, {1,      ly - 1}},   // middle block
            {{0,      1},      {1,      ly - 1}},        // left col no corners
            {{lx - 1, lx},     {1,      ly - 1}},  // right col no corners
            {{1,      lx - 1}, {0,      1}},        // top row no corners <--
            {{1,      lx - 1}, {ly - 1, ly}},  // bottom row no corners
            {{0,      1},      {0,      1}},             // top left cell
            {{0,      1},      {ly - 1, ly}},       // bottom left cell
            {{lx - 1, lx},     {0,      1}},       // top right cell
            {{lx - 1, lx},     {ly - 1, ly}}, // bottom right cell
    };

    auto dst_slices = std::vector<SliceRegion>();
    std::transform(src_slices.begin(), src_slices.end(), std::back_inserter(dst_slices),
                   [dims, roll_offsets](SliceRegion i) -> SliceRegion { return toDstSlices(dims, roll_offsets, i); });
    return {.src = src_slices, .dst = dst_slices};
}

auto doubleRolledCopy(Graph &g, Tensor &src, Tensor &dst, int roll_x, int roll_y) -> Program {
//    std::cerr << __FUNCTION__ << std::endl;

    assert(roll_x > -2 && roll_x < 2);
    assert(roll_y > -2 && roll_y < 2);

    auto s = Sequence();

    const auto &[src_slices, dst_slices] = determineSrcAndDstSlices({src.dim(0), src.dim(1)}, {roll_x, roll_y});

    for (auto i = 0; i < 9; i++) {
        auto const &[src_xSlice, src_ySlice] = src_slices[i];
        auto const &[src_xFrom, src_xTo] = src_xSlice;
        auto const &[src_yFrom, src_yTo] = src_ySlice;
        auto const &[dst_xSlice, dst_ySlice] = dst_slices[i];
        auto const &[dst_xFrom, dst_xTo] = dst_xSlice;
        auto const &[dst_yFrom, dst_yTo] = dst_ySlice;
//        std::cout << "src[[" << src_xFrom << ":" << src_xTo << "),[" << src_yFrom << ":" << src_yTo << ")]"
//                  << std::endl;
//        std::cout << "dst[[" << dst_xFrom << ":" << dst_xTo << "),[" << dst_yFrom << ":" << dst_yTo << ")]"
//                  << std::endl;
//        std::cerr << "i " << i << " " << src_xFrom << " " << src_xTo << " " << src_yFrom << " " << src_yTo << " "
//                  << dst_xFrom << " " << dst_xTo << " " << dst_yFrom << " " << dst_yTo << " " << __LINE__ << std::endl;
//        std::cerr << src.dim(0) << "x" << src.dim(1) << std::endl;
        auto src_slice = src.slice({src_xFrom, src_yFrom}, {src_xTo, src_yTo});
        auto dst_slice = dst.slice({dst_xFrom, dst_yFrom}, {dst_xTo, dst_yTo});
        s.add(Copy(src_slice, dst_slice));
    }
//    std::cerr << "done" << __FUNCTION__ << std::endl;

    return std::move(s);
}

#endif //LBM_GRAPHCORE_DOUBLEROLL_H
