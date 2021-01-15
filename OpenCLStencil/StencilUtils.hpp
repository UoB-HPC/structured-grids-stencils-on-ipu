#ifndef STENCIL_UTILS_H
#define STENCIL_UTILS_H

#include <algorithm>
#include <limits>
#include "lib/lodepng/lodepng.h"
#include <stdexcept>
#include <iostream>

namespace stencil {
    constexpr auto NumChannels = 4u; // RGBA

    typedef struct {
        unsigned width;
        unsigned height;
        std::vector<unsigned char> bytes; // Note: this is in row, col, channel order
    } Image;

    typedef struct {
        unsigned width;
        unsigned height;
        unsigned origChanMin[NumChannels]; // Note the original image's min and max values per chan so that we can reconstruct an image with the same brightness
        unsigned origChanMax[NumChannels];
        std::vector<float> intensities;
    } FloatImage;

    auto duplicate(FloatImage &in) -> FloatImage {
        auto out = FloatImage();
        out.width = in.width;
        out.height = in.height;
        for (auto c = 0u; c < NumChannels; c++) {
            out.origChanMax[c] = in.origChanMax[c];
            out.origChanMin[c] = in.origChanMin[c];
        }
        out.intensities = std::vector<float>(in.intensities);
        return out;
    }

    auto loadPng(const std::string &filename) -> Image {
        auto img = Image{};

        unsigned error = lodepng::decode(img.bytes, img.width, img.height, filename);

        if (error) {
            std::cerr << "PNG decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	    throw std::runtime_error("Could not decode PNG file");
        }

        return img;
    }

    auto savePng(const Image &image, const std::string &filename) -> bool {
        unsigned error = lodepng::encode(filename, image.bytes, image.width, image.height);

        if (error) std::cerr << "PNG encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        return error == 0;
    }

    // Assumes a channels-last image
    auto zeroPad(const FloatImage &imageChannelsLast) -> FloatImage {
        auto newImage = FloatImage{
                .width= imageChannelsLast.width + 2,
                .height= imageChannelsLast.height + 2,
                .origChanMin = {imageChannelsLast.origChanMin[0], imageChannelsLast.origChanMin[1],
                                imageChannelsLast.origChanMin[2], imageChannelsLast.origChanMin[3]},
                .origChanMax = {imageChannelsLast.origChanMax[0], imageChannelsLast.origChanMax[1],
                                imageChannelsLast.origChanMax[2], imageChannelsLast.origChanMax[3]},
                .intensities = std::vector<float>(
                        NumChannels * (imageChannelsLast.height + 2) * (imageChannelsLast.width + 2), 0.f)
        };
#pragma omp parallel for  default(none) shared( newImage, imageChannelsLast)  schedule(static, 4) collapse(3)
        for (auto row = 0u; row < newImage.height; row++) {
            for (auto col = 0u; col < newImage.width; col++) {
                for (auto chan = 0u; chan < NumChannels; chan++) {
                    const auto idx = (row * newImage.width + col) * NumChannels + chan;
                    if (row == 0 || col == 0 || row == imageChannelsLast.height + 1 ||
                        col == imageChannelsLast.width + 1) {
                        newImage.intensities[idx] = 0.f;
                    } else {
                        const auto origIdx =
                                (row - 1) * ((imageChannelsLast.width) * NumChannels) + (col - 1) * NumChannels + chan;
                        newImage.intensities[idx] = imageChannelsLast.intensities[origIdx];
                    }
                }
            }
        }
        return newImage;
    }

    // Assumes a channels-last image
    auto stripPadding(const FloatImage &paddedImage) -> FloatImage {
        auto newImage = FloatImage{
                .width= paddedImage.width - 2,
                .height= paddedImage.height - 2,
                .origChanMin = {paddedImage.origChanMin[0], paddedImage.origChanMin[1],
                                paddedImage.origChanMin[2], paddedImage.origChanMin[3]},
                .origChanMax = {paddedImage.origChanMax[0], paddedImage.origChanMax[1],
                                paddedImage.origChanMax[2], paddedImage.origChanMax[3]},
                .intensities = std::vector<float>(NumChannels * (paddedImage.height - 2) * (paddedImage.width - 2), 0.f)
        };
#pragma omp parallel for  default(none) shared(newImage, paddedImage)  schedule(static, 4) collapse(3)
        for (auto row = 1u; row < paddedImage.height - 1; row++) {
            for (auto col = 1u; col < paddedImage.width - 1; col++) {
                for (auto chan = 0u; chan < NumChannels; chan++) {
                    const auto idx = (row - 1) * ((newImage.width) * NumChannels) + (col - 1) * NumChannels + chan;
                    const auto origIdx = row * ((paddedImage.width) * NumChannels) + col * NumChannels + chan;
                    newImage.intensities[idx] = paddedImage.intensities[origIdx];
                }
            }
        }
        return newImage;
    }

    auto toChannelsFirst(const FloatImage &imageChannelsLast) -> FloatImage {
        auto newImage = FloatImage{
                .width= imageChannelsLast.width,
                .height= imageChannelsLast.height,
                .origChanMin = {imageChannelsLast.origChanMin[0], imageChannelsLast.origChanMin[1],
                                imageChannelsLast.origChanMin[2], imageChannelsLast.origChanMin[3]},
                .origChanMax = {imageChannelsLast.origChanMax[0], imageChannelsLast.origChanMax[1],
                                imageChannelsLast.origChanMax[2], imageChannelsLast.origChanMax[3]},
                .intensities = std::vector<float>(NumChannels * imageChannelsLast.height * imageChannelsLast.width, 0.f)
        };
#pragma omp parallel for  default(none) shared(newImage, imageChannelsLast)  schedule(static, 4) collapse(3)
        for (auto row = 0u; row < newImage.height; row++) {
            for (auto col = 1u; col < newImage.width; col++) {
                for (auto chan = 0u; chan < NumChannels; chan++) {
                    const auto chanLastIdx = row * (newImage.width * NumChannels) + col * NumChannels + chan;
                    const auto chanFirstIdx = chan * (newImage.height * newImage.width) + row * newImage.width + col;
                    newImage.intensities[chanFirstIdx] = imageChannelsLast.intensities[chanLastIdx];
                }
            }
        }
        return newImage;
    }

    auto toChannelsLast(const FloatImage &channelsFirst) -> FloatImage {
        auto newImage = FloatImage{
                .width= channelsFirst.width,
                .height= channelsFirst.height,
                .origChanMin = {channelsFirst.origChanMin[0], channelsFirst.origChanMin[1],
                                channelsFirst.origChanMin[2], channelsFirst.origChanMin[3]},
                .origChanMax = {channelsFirst.origChanMax[0], channelsFirst.origChanMax[1],
                                channelsFirst.origChanMax[2], channelsFirst.origChanMax[3]},
                .intensities = std::vector<float>(NumChannels * channelsFirst.height * channelsFirst.width, 0.f)
        };
#pragma omp parallel for  default(none) shared(newImage, channelsFirst)  schedule(static, 4) collapse(3)
        for (auto row = 0u; row < newImage.height; row++) {
            for (auto col = 0u; col < newImage.width; col++) {
                for (auto chan = 0u; chan < NumChannels; chan++) {
                    const auto chanLastIdx = row * (newImage.width * NumChannels) + col * NumChannels + chan;
                    const auto chanFirstIdx = chan * (newImage.height * newImage.width) + row * newImage.width + col;
                    newImage.intensities[chanLastIdx] = channelsFirst.intensities[chanFirstIdx];
                }
            }
        }
        return newImage;
    }

// fills a channel-last float version of the input image, with pixels in the range 0..1 and original image intensity ranges stored
    auto toFloatImage(const Image &image) -> FloatImage {
        auto fImg = FloatImage{
                .width= image.width,
                .height= image.height,
                .origChanMin = {0, 0, 0, 0},
                .origChanMax = {0, 0, 0, 0},
                .intensities = std::vector<float>(NumChannels * image.height * image.width, 0.f)
        };
        unsigned char max[4] = {0, 0, 0, 0};
        constexpr auto MaxUChar = std::numeric_limits<unsigned char>::max();
        unsigned char min[4] = {MaxUChar, MaxUChar, MaxUChar, MaxUChar};
        for (auto idx = 0u; idx < image.width * image.height; idx++) {
            const auto rgba = &image.bytes[idx * NumChannels];
            for (auto chan = 0u; chan < NumChannels; chan++) {
                max[chan] = std::max(max[chan], rgba[chan]);
                min[chan] = std::min(min[chan], rgba[chan]);
            }
        }
        for (auto c = 0u; c < NumChannels; c++) {
            fImg.origChanMin[c] = min[c];
            fImg.origChanMax[c] = max[c];
#pragma omp parallel for default(none) shared(max, min, image, fImg, c)  schedule(static, 4) collapse(2)
            for (auto y = 0u; y < image.height; y++) {
                for (auto x = 0u; x < image.width; x++) {
                    const auto idx = (y * image.width + x) * NumChannels + c;
                    fImg.intensities[idx] = (max[c] == min[c])
                                            ? 0.f :
                                            ((float) image.bytes[idx] - (float) min[c]) /
                                            ((float) max[c] - (float) min[c]);
                }
            }
        }

        return fImg;
    }

// Creates a channel-last char version of the input image (assumed channels-last), rescaling the intensity ranges from the given ImageDescriptor
    auto toCharImage(const FloatImage &floatImage) -> Image {
        auto img = Image{
                .width= floatImage.width,
                .height= floatImage.height,
                .bytes= std::vector<unsigned char>(NumChannels * floatImage.height * floatImage.width, 0)
        };

        // Find the min and max vals of each channel
        float max[4] = {0.f, 0.f, 0.f, 0.f};
        constexpr auto MaxFloat = std::numeric_limits<float>::max();
        float min[4] = {MaxFloat, MaxFloat, MaxFloat, MaxFloat};

#pragma omp parallel for  default(none) shared(max, min, floatImage)  schedule(static, 4) collapse(3)
        for (auto c = 0u; c < NumChannels; c++) {
            for (auto y = 0u; y < floatImage.height; y++) {
                for (auto x = 0u; x < floatImage.width; x++) {
                    const auto idx = c * (floatImage.height * floatImage.width) + y * floatImage.width + x;
                    min[c] = std::min(min[c], floatImage.intensities[idx]);
                    max[c] = std::max(max[c], floatImage.intensities[idx]);
                }
            }
        }
#pragma omp parallel for  default(none) shared(max, min, img, floatImage)  schedule(static, 4) collapse(3)
        for (auto c = 0u; c < NumChannels; c++) {
            for (auto y = 0u; y < img.height; y++) {
                for (auto x = 0u; x < img.width; x++) {
                    const auto idx = (y * img.width + x) * NumChannels + c;
                    auto rescaled = (max[c] == min[c]) ? 0.f :
                                    (floatImage.intensities[idx] + min[c]) /
                                    (max[c] - min[c]); // Now it's in the range 0..1
                    auto inOrigBrightness =
                            (rescaled * (float) (floatImage.origChanMax[c] - floatImage.origChanMin[c])) +
                            (float) floatImage.origChanMin[c];
                    if (inOrigBrightness > floatImage.origChanMax[c]) {
                        inOrigBrightness = floatImage.origChanMax[c];
                    }
                    img.bytes[idx] = (unsigned char) inOrigBrightness;
                }
            }
        }

        return img;
    }

}
#endif
