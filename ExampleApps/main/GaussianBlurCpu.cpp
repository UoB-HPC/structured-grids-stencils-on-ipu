#include <cstdlib>
#include <cxxopts.hpp>
#include <lodepng.h>
#include "include/StructuredGridUtils.hpp"
#include <chrono>
#include "include/ImageUtils.hpp"

using namespace stencil;

// in and out are 4 channels of float pixel data stored in chan, row, col order
// These are the zero-padded arrays
auto gaussian_blur(const float *in, float *out, const unsigned nx, const unsigned ny) {
    for (auto y = 1u; y < ny - 1; y++) {
        for (auto x = 1u; x < nx - 1; x++) {
            const auto idx = y * nx + x;
            const int up = -nx;
            const int down = +nx;
            constexpr int left = -1;
            constexpr int right = 1;
            const auto nw = idx + up + left;
            const auto ne = idx + up + right;
            const auto n = idx + up;
            const auto w = idx + left;
            const auto m = idx;
            const auto e = idx + right;
            const auto sw = idx + down + left;
            const auto s = idx + down;
            const auto se = idx + down + right;
            out[m] =
                    (2.f * (in[n] + in[s] + in[w] + in[e]) + 1.f * (in[nw] + in[ne] + in[sw] + in[se]) + 4.0f * in[m]) /
                    16.f;
        }
    }
}


int main(int argc, char *argv[]) {
    std::string inputFilename, outputFilename;
    unsigned numIters = 100u;

    cxxopts::Options options(argv[0], " - Runs a 3x3 Gaussian Blur stencil over an input image a number of times");
    options.add_options()
            ("n,num-iters", "Number of iterations", cxxopts::value<unsigned>(numIters)->default_value("1"))
            ("i,image", "filename input image (must be a 4-channel PNG image)",
             cxxopts::value<std::string>(inputFilename))
            ("o,output", "filename output (blurred image)", cxxopts::value<std::string>(outputFilename));
    auto opts = options.parse(argc, argv);

    try {
        auto opts = options.parse(argc, argv);
        if (!(inputFilename.length() && outputFilename.length())) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    } catch (cxxopts::OptionParseException &) {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    auto maybeImg = loadPng(inputFilename);
    if (!maybeImg.has_value()) {
        return EXIT_FAILURE;
    }

    cout << inputFilename << " is " << maybeImg->width << "x" << maybeImg->height << " pixels in size." << std::endl;

    alignas(64) auto tmp_img = std::make_unique<float[]>((maybeImg->width + 2) * (maybeImg->height + 2) * NumChannels);
    auto fImage = toChannelsFirst(zeroPad(toFloatImage(*maybeImg)));
    auto img = fImage.intensities.data();

    numIters = 100;
    std::cout << "Running " << numIters << "(x2) iterations of stencil";
    auto tic = std::chrono::high_resolution_clock::now();

    for (auto iter = 0u; iter < numIters; iter++) {
        for (auto c = 0u; c < NumChannels-1; c++) {
            const auto offset = c * fImage.width * fImage.height;
            gaussian_blur(&img[offset], &tmp_img[offset], fImage.width, fImage.height);
            gaussian_blur(&tmp_img[offset], &img[offset], fImage.width, fImage.height);
        }
    }

    auto toc = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
    std::cerr << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;

    auto cImg = toCharImage(stripPadding(toChannelsLast(fImage)));

    if (!savePng(cImg, outputFilename)) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
