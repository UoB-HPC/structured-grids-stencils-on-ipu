
#include <cstdlib>
#include <cxxopts.hpp>
#include "include/StructuredGridUtils.hpp"
#include <chrono>
#include <iostream>
#include <sstream>

int main(int argc, char *argv[]) {
    std::string outputFilename;
    unsigned numIpus = 1u;
    unsigned minWidth = 128u;
    unsigned minHeight = 128u;
    unsigned maxWidth = 4000u;
    unsigned maxHeight = 4000u;
    unsigned numSamples = 10000u;
    unsigned minRowsPerTile = 6u;
    unsigned minColsPerTile = 6u;

    cxxopts::Options options(argv[0],
                             " - Work out the stats for a range of matrix sizes after they have been mapped");
    options.add_options()
            ("num-ipus", "Number of IPUs to target (1,2,4,8 or 16)", cxxopts::value<unsigned>(numIpus))
            ("min-width", "Min width", cxxopts::value<unsigned>(minWidth))
            ("max-width", "Max width", cxxopts::value<unsigned>(maxWidth))
            ("min-height", "Min height", cxxopts::value<unsigned>(minHeight))
            ("min-rows-per-tile", "Min rows per tile (default 6)",
             cxxopts::value<unsigned>(minRowsPerTile)->default_value(std::to_string(grids::DefaultMinRowsPerTile)))
            ("min-cols-per-tile", "Min cols per tile (default 6)",
             cxxopts::value<unsigned>(minColsPerTile)->default_value(std::to_string(grids::DefaultMinColsPerTile)))
            ("n,num-samples", "Number of samples", cxxopts::value<unsigned>(numSamples)->default_value("100000"))
            ("max-height", "Max height", cxxopts::value<unsigned>(maxHeight));
    bool sample = false;
    try {
        auto opts = options.parse(argc, argv);
        sample = opts.count("num-samples") > 0;
        if (opts.count("min-width") + opts.count("min-height") + opts.count("max-width") + opts.count("max-height") <
            4) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    } catch (cxxopts::OptionParseException &) {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    constexpr auto NumTiles = 1216;
    constexpr auto NumWorkers = 6;

    std::cout << "numIpus,width,height,wastedTiles,wastedWorkers,loadBalance,maxSpeedup,aveBlockSize,largestBlockSize,aveTileBlockSize,largestTileBlockSize,smallestTileBlockSize"
              << std::endl;


    auto doSample = [&](int height, int width) -> bool {
        auto ipuLevelMappings = grids::partitionForIpus(
                {(unsigned) height, (unsigned) width}, numIpus,
                (unsigned) std::min(4000 * 4000.f, (maxHeight * maxWidth) / (float) numIpus));
        if (!ipuLevelMappings.has_value()) { // we can't fit this size onto the IPU
            return false;
        }
        auto tileLevelMappings = grids::toTilePartitions(*ipuLevelMappings, NumTiles, minRowsPerTile, minColsPerTile);
        auto workerLevelMappings = grids::toWorkerPartitions(tileLevelMappings);

        // what size of the input area a '100% busy worker' will be
        auto busiestWorker = 0u;
        for (const auto &[target, slice]: workerLevelMappings) {
            const auto busyness = slice.width() * slice.height();
            if (busyness > busiestWorker) busiestWorker = busyness;
        }

        auto busiestTile = 0u;
        auto smallestTileBlockSize = 1000*1000;
        for (const auto &[target, slice]: tileLevelMappings) {
            const auto busyness = slice.width() * slice.height();
            if (busyness > busiestTile) busiestTile = busyness;
            if (busyness < smallestTileBlockSize) smallestTileBlockSize = busyness;
        }

        double loadBalanceTotal = 0.0;
        double wasteWorkers = NumTiles * numIpus * NumWorkers - workerLevelMappings.size();
        double wasteTiles = NumTiles * numIpus - tileLevelMappings.size();
        for (const auto &[target, slice]: workerLevelMappings) {
            auto workerLoad = slice.width() * slice.height();
            loadBalanceTotal += workerLoad;
        }
        double aveLoadBalance = (loadBalanceTotal / workerLevelMappings.size()) / busiestWorker * 100;
        assert(loadBalanceTotal == height * width);
        const double aveWorkerBlockSize = height * width / workerLevelMappings.size();
        const double aveTileBlockSize = height * width / tileLevelMappings.size();
        const double largestWorkerBlockSize = busiestWorker;
        const double largestTileBlockSize = busiestTile;

        double maxSpeedup = ((double) width * height) / ((double) busiestWorker);
        std::cout << numIpus << "," << (int) width << "," << (int) height << "," << wasteTiles << "," << wasteWorkers
                  << ","
                  << aveLoadBalance << "," << maxSpeedup << "," << aveWorkerBlockSize << "," << largestWorkerBlockSize
                  <<
                  "," << aveTileBlockSize << "," << largestTileBlockSize << "," << smallestTileBlockSize << std::endl;

        return true;
    };

    if (sample) {
        auto sampleNum = 0u;
        while (sampleNum < numSamples) {
            auto height = rand() % (maxHeight - minHeight + 1) + minHeight;
            auto width = rand() % (maxWidth - minWidth + 1) + minWidth;
            if (doSample(height, width)) {
                sampleNum++;
            }
        }
    } else {
        for (double height = minHeight; height <= maxHeight; height += ((maxHeight - minHeight) / 200.0)) {
            for (double width = minWidth; width <= maxWidth; width += ((maxWidth - minWidth) / 200.0)) {
                doSample(height, width);
            }
        }
    }


    return EXIT_SUCCESS;
}