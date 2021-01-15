
#include <cstdlib>
#include <cxxopts.hpp>
#include <lodepng.h>
#include "include/StructuredGridUtils.hpp"
#include <chrono>
#include "include/ImageUtils.hpp"
#include <iostream>
#include <sstream>

using namespace stencil;

class Colour {
public:
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    unsigned char alpha;

    Colour(const unsigned char r, const unsigned char g, const unsigned char b, const unsigned char a = 0xFF) : red(r),
                                                                                                                green(g),
                                                                                                                blue(b),
                                                                                                                alpha(a) {}
};

const auto setPixel(const unsigned x, const unsigned y, const Colour c, Image &image) -> void {
    const auto idx = (y * image.width + x) * NumChannels;
    image.bytes[idx + 0] = c.red;
    image.bytes[idx + 1] = c.green;
    image.bytes[idx + 2] = c.blue;
    image.bytes[idx + 3] = c.alpha;
};

auto drawBox(const unsigned x, const unsigned y, const unsigned width, const unsigned height, const Colour c,
             Image &image) -> void {
    const auto top = y;
    const auto bottom = y + height - 1;
    const auto left = x;
    const auto right = x + width - 1;

    for (auto x = left; x <= right; x++) {
        setPixel(x, top, c, image);
        setPixel(x, bottom, c, image);
    }
    for (auto y = top + 1; y <= bottom - 1; y++) {
        setPixel(left, y, c, image);
        setPixel(right, y, c, image);
    }
}

auto fillBox(const unsigned x, const unsigned y, const unsigned width, const unsigned height, const Colour c,
             Image &image) -> void {
    const auto top = y;
    const auto bottom = y + height - 1;
    const auto left = x;
    const auto right = x + width - 1;

    for (auto y = top; y <= bottom; y++) {
        for (auto x = left; x <= right; x++) {
            setPixel(x, y, c, image);
        }
    }
}


int main(int argc, char *argv[]) {
    std::string outputFilename;
    unsigned numIpus = 1u;
    unsigned width = 128u;
    unsigned height = 128u;

    cxxopts::Options options(argv[0],
                             " - Show how a grid of the given size would be mapped to the IPU(s)");
    options.add_options()
            ("num-ipus", "Number of IPUs to target (1,2,4,8 or 16)", cxxopts::value<unsigned>(numIpus))
            ("o,output", "filename output (blurred image)", cxxopts::value<std::string>(outputFilename))
            ("width", "Width of input", cxxopts::value<unsigned>(width))
            ("height", "Hieght of input", cxxopts::value<unsigned>(height));

    try {
        auto opts = options.parse(argc, argv);
        if (opts.count("width") + opts.count("height") + opts.count("o") < 3) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    } catch (cxxopts::OptionParseException &) {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    cout << "Visualising a grid that is " << height << " rows by " << width << " cols in size." << std::endl;

    constexpr auto IpuBorder = 4;
    constexpr auto TileBorder = 1;
    constexpr auto WorkerWidth = 12;
    constexpr auto WorkerHeight = 2;
    constexpr auto NumTilesInRowPerIpu = 32;
    constexpr auto NumTilesInColPerIpu = 38;
    constexpr auto NumTiles = 1216;
    constexpr auto NumWorkers = 6;
    constexpr auto TileHeight = NumWorkers * WorkerHeight;
    constexpr auto OneIpuWidth = 32 * (WorkerWidth + TileBorder) + TileBorder;
    constexpr auto OneIpuHeight = 38 * (TileHeight + TileBorder) + TileBorder;
    const auto RowsOfIpus = std::ceil((float) numIpus / 4.0f);
    const auto ColsOfIpus = std::min((float) numIpus, 4.0f);
    const auto ImageWidth = (unsigned) ColsOfIpus * (OneIpuWidth + IpuBorder) + IpuBorder;
    const auto ImageHeight = (unsigned) RowsOfIpus * (OneIpuHeight + IpuBorder) + IpuBorder;


    auto ipuLevelMappings = grids::partitionForIpus({height, width}, numIpus, 4000 * 4000);
    if (!ipuLevelMappings.has_value()) {
        std::cerr << "Couldn't fit the problem on the " << numIpus << " ipus." << std::endl;
        return EXIT_FAILURE;
    }
    auto tileLevelMappings = grids::toTilePartitions(*ipuLevelMappings, NumTiles);
    auto workerLevelMappings = grids::toWorkerPartitions(tileLevelMappings);

    auto colours = std::array<Colour, 7>{Colour{0x23, 0x25, 0x42},
                                         Colour{0x72, 0x1D, 0x3A},
                                         Colour{0xf4, 0xa3, 0x16},
                                         Colour{0x4a, 0x5b, 0xe3},
                                         Colour{0x3C, 0x2a, 0x27},
                                         Colour{0x77, 0x32, 0x27},
                                         Colour{0x00, 0x00, 0x00, 0x33}};


    // what size of the input area a '100% busy worker' will be

    auto busiestWorker = 0u;
    for (const auto &[target, slice]: workerLevelMappings) {
        const auto busyness = slice.width() * slice.height();
        if (busyness > busiestWorker) busiestWorker = busyness;
    }


    stencil::Image image = {
            .width = ImageWidth,
            .height = ImageHeight,
            .bytes = std::vector<unsigned char>(ImageWidth * ImageHeight * NumChannels, 0xFF)
    };

    // Borders for the IPUs
    for (auto ipu = 0u; ipu < numIpus; ipu++) {
        const auto ipuRow = ipu / 4;
        const auto ipuCol = ipu % 4;
        auto x = ipuCol * OneIpuWidth + IpuBorder * (ipuCol + 1);
        auto y = ipuRow * OneIpuHeight + IpuBorder * (ipuRow + 1);
        for (auto b = 1; b <= IpuBorder; b++) {
            drawBox(x - b, y - b, OneIpuWidth + b + b, OneIpuHeight + b + b, colours[6],
                    image);
        }
    }

    // Borders for the Tiles
    for (auto ipu = 0u; ipu < numIpus; ipu++) {
        for (auto tile = 0u; tile < NumTiles; tile++) {
            const auto ipuRow = ipu / 4;
            const auto ipuCol = ipu % 4;
            const auto tileRow = tile / NumTilesInRowPerIpu;
            const auto tileCol = tile % NumTilesInRowPerIpu;
            auto ipuX = ipuCol * OneIpuWidth + IpuBorder * (ipuCol + 1);
            auto ipuY = ipuRow * OneIpuHeight + IpuBorder * (ipuRow + 1);
            auto tileX = tileCol * WorkerWidth + TileBorder * (tileCol + 1);
            auto tileY = tileRow * TileHeight + TileBorder * (tileRow + 1);
            auto x = ipuX + tileX;
            auto y = ipuY + tileY;
            for (auto b = 1; b <= TileBorder; b++) {
                drawBox(x - b, y - b, WorkerWidth + TileBorder + b, TileHeight + TileBorder + b, colours[6],
                        image);
            }
        }
    }

    double loadBalanceTotal = 0.0;
    double wasteWorkers = NumTiles * numIpus * NumWorkers - workerLevelMappings.size();
    double wasteTiles = NumTiles * numIpus - tileLevelMappings.size();
    for (const auto &[target, slice]: workerLevelMappings) {
        const auto ipuRow = target.ipu() / 4;
        const auto ipuCol = target.ipu() % 4;
        const auto tileRow = target.tile() / NumTilesInRowPerIpu;
        const auto tileCol = target.tile() % NumTilesInRowPerIpu;
        const auto worker = target.worker();
        auto ipuX = ipuCol * OneIpuWidth + IpuBorder * (ipuCol + 1);
        auto ipuY = ipuRow * OneIpuHeight + IpuBorder * (ipuRow + 1);
        auto tileX = tileCol * WorkerWidth + TileBorder * (tileCol + 1);
        auto tileY = tileRow * TileHeight + TileBorder * (tileRow + 1);
        auto x = ipuX + tileX;
        auto y = ipuY + tileY + worker * WorkerHeight;
        auto c = colours[worker];
        auto workerLoad  =slice.width() * slice.height();
        loadBalanceTotal += workerLoad;
        auto width = (int) (((float) workerLoad / (float) busiestWorker) * WorkerWidth);
        fillBox(x, y,width , WorkerHeight, c, image);
    }
    double aveLoadBalance = (loadBalanceTotal / workerLevelMappings.size()) / busiestWorker * 100;
    std::cout << "Load balance:" << aveLoadBalance << std::endl;
    std::cout << "Wasted tiles:" << wasteTiles << std::endl;
    std::cout << "Wasted workers:" << wasteWorkers << std::endl;
    std::cout << "Max embarassingly parallel speedup:" << ((double) width * height)/((double) busiestWorker) << std::endl;

    if (!savePng(image, outputFilename)) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}