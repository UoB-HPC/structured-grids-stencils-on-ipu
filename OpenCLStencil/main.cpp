#include <iostream>
#include <chrono>
#include <iomanip>
#include <cassert>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include "CL/cl2.hpp"
#include "StencilUtils.hpp"
#include "lib/cxxopts/cxxopts.hpp"
#include "lib/half-2.1.0/include/half.hpp"


auto getDeviceList() -> std::vector<cl::Device> {
    auto devices = std::vector<cl::Device>{};
    auto platforms = std::vector<cl::Platform>{};
    cl::Platform::get(&platforms);

    for (auto &platform : platforms) {
        auto plat_devices = std::vector<cl::Device>{};
        platform.getDevices(CL_DEVICE_TYPE_ALL, &plat_devices);
        devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
    }
    return devices;
}

auto getDeviceName(std::vector<cl::Device> devices, const size_t device) -> std::string {
    assert(device < devices.size());
    std::string name;
    devices[device].getInfo(CL_DEVICE_NAME, &name);
    return name;
}

auto getDeviceDriver(std::vector<cl::Device> devices, const size_t device) -> std::string {
    assert(device < devices.size());
    std::string driver;
    devices[device].getInfo(CL_DRIVER_VERSION, &driver);
    return driver;
}

auto listDevices() -> void {
	auto devices =  getDeviceList();
    if (devices.empty()) {
        std::cerr << "No devices found." << std::endl;
    } else {
        std::cout << std::endl << "Devices:" << std::endl;
        for (auto i = 0u; i < devices.size(); i++) {
            std::cout << i << ": " << getDeviceName(devices, i) << std::endl;
        }
        std::cout << std::endl;
    }
}

constexpr auto kernel_float = R"CLC(
#define OFFSET(r,c) (y + r) * nx + (x + c)
__kernel void gaussian_blur(
        __global __read_only const float4  *in ,
        __global __write_only float4  *out,
        const unsigned nx,
        const unsigned ny
    ) {

        const unsigned x = get_global_id(0);
        const unsigned y = get_global_id(1);

        out[OFFSET(0,0)] = in[OFFSET(0,0)] * 0.25f +
             (in[OFFSET(-1,-1)] + in[OFFSET(-1,1)] + in[OFFSET(1,1)] + in[OFFSET(1,-1)]) * 0.0625f +
             (in[OFFSET(-1,0)] + in[OFFSET(1,0)] + in[OFFSET(0,1)] + in[OFFSET(0,-1)]) * 0.125f;

        return;
    }

)CLC";


constexpr auto kernel_half = R"CLC(
#define OFFSET(r,c) (y + r) * nx + (x + c)
__kernel void gaussian_blur(
        __global __read_only const half4  *in ,
        __global __write_only half4  *out,
        const unsigned nx,
        const unsigned ny
) {

    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);

    out[OFFSET(0,0)] = in[OFFSET(0,0)] * 0.25f +
                       (in[OFFSET(-1,-1)] + in[OFFSET(-1,1)] + in[OFFSET(1,1)] + in[OFFSET(1,-1)]) * 0.0625f +
                       (in[OFFSET(-1,0)] + in[OFFSET(1,0)] + in[OFFSET(0,1)] + in[OFFSET(0,-1)]) * 0.125f;

    return;
}

)CLC";

constexpr auto kernel_float_optim = R"CLC(
// Offset in global buffer
 #define OFFSET(r,c) (y + r) * nx + (x + c)

 // Offset in local buffer
#define LOCAL_OFFSET(r,c) (local_y + r) * group_size_x + c + local_x

 // Whether or not this offset is in local memory (or we must read from global)
 #define IN_TILE(R,C) ((x + C) >= tile_start_x && (x + C) <= tile_end_x && (y + R) >= tile_start_y && (y + R) <= tile_end_y)

// Neighbour at the given offset, read from the appropriate store (local if in tile, global if halo)
#define NEIGHBOUR(R,C) (IN_TILE(R,C) ?  in_local[LOCAL_OFFSET(R,C)]: in[OFFSET(R,C)])
__kernel void gaussian_blur(
        __global __read_only const float4  *in ,
	__global __write_only float4  *out,
	const unsigned nx,
	const unsigned ny
	    ) {
	    const int local_x = get_local_id(0);
	    const int local_y = get_local_id(1);
	    const int group_x = get_group_id(0);
	    const int group_y = get_group_id(1);
	    const int group_size_x = get_local_size(0);
	    const int group_size_y = get_local_size(1);
                                                                                            
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int tile_start_x = group_x*group_size_x;
	const int tile_end_x = (group_x+1)*group_size_x-1;
	const int tile_start_y = group_y*group_size_y;
	const int tile_end_y = (group_y+1)*group_size_y-1;

	__local float4 in_local[TILE_SIZE];
	in_local[LOCAL_OFFSET(0,0)] = in[OFFSET(0,0)];  // Write our data to the local tile buffer
	barrier(CLK_LOCAL_MEM_FENCE);

	// Need to figure out whether to read from local or global
	out[OFFSET(0,0)] = NEIGHBOUR(0,0) * 0.25f+
		0.0625f * (NEIGHBOUR(-1,-1) + NEIGHBOUR(1,1) +NEIGHBOUR(1,1) + NEIGHBOUR(1,-1)) +
	    0.125f  * (NEIGHBOUR(0,1) + NEIGHBOUR(0,-1) + NEIGHBOUR(1,0) + NEIGHBOUR(1,0));
																									}
)CLC";



int main(int argc, char *argv[]) {
    std::string inputFilename, outputFilename;
    unsigned deviceNum = 0u;
    unsigned numIters = 100u;
    std::string dataType = "float";

    cxxopts::Options options(argv[0],
                             " - Runs a 3x3 Gaussian Blur stencil over an input image a number of times using an OpenCL device");
    options.add_options()
            ("n,num-iters", "Number of iterations", cxxopts::value<unsigned>(numIters)->default_value("1"))
            ("i,image", "filename input image (must be a 4-channel PNG image)",
             cxxopts::value<std::string>(inputFilename))
            ("data-type", "Data type (float, half)",
             cxxopts::value<std::string>(dataType)->default_value("float"))
            ("d,device", "Device number to use",
             cxxopts::value<unsigned>(deviceNum)->default_value("0"))
            ("l,list-devices", "If set, just lists the OpenCL devices and their numbers and quits")
            ("o,output", "filename output (blurred image)", cxxopts::value<std::string>(outputFilename));
    try {
        auto opts = options.parse(argc, argv);
        if (opts["list-devices"].as<bool>()) {
            listDevices();
            return EXIT_SUCCESS;
        }
        if (opts.count("n") + opts.count("i") + opts.count("o") + opts.count("d") < 4) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    } catch (cxxopts::OptionParseException &) {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    auto maybeImg = stencil::loadPng(inputFilename);
    std::cout << inputFilename << " is " << maybeImg.width << "x" << maybeImg.height << " pixels in size."
              << std::endl;

    // OpenCL objects
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

    auto devices = getDeviceList();
    device = devices[deviceNum];

    std::cout << "Using data type " << dataType << std::endl;

    if (dataType == "half") {
        bool ok = false;
        try {
            if (device.getInfo<CL_DEVICE_HALF_FP_CONFIG>())
                ok = true;
        } catch (cl::Error &err) {
        }
        if (!ok) {
            std::cerr << "Device does not support half precision, please use --data-type float" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Print out device information
    std::cout << "Using OpenCL device " << getDeviceName(devices, deviceNum) << std::endl;
    std::cout << "Driver: " << getDeviceDriver(devices, deviceNum) << std::endl;

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    cl::Program program(context, dataType == "half" ? kernel_half : kernel_float_optim);

    const auto TILE_WIDTH = 2;
    const auto TILE_HEIGHT = 11u;
    try {
        std::stringstream opts;
        opts << "-cl-fast-relaxed-math ";
        opts << "-DTILE_SIZE=" << TILE_WIDTH * TILE_HEIGHT;
        program.build(opts.str().c_str());
    }
    catch (cl::Error &err) {
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()[0].second << std::endl;
            throw err;
        }
    }
    const auto typeSize = dataType == "half" ? sizeof(uint16_t) : sizeof(float);
    auto image = stencil::zeroPad(stencil::toFloatImage(maybeImg));



    auto float16DataBuf = std::vector<uint16_t>{};
    if (dataType == "half") {
        const auto height = image.height;
        const auto width = image.width;
        float16DataBuf.resize(width * height * stencil::NumChannels);

        for (auto y = 0u; y < height; y++) {
            for (auto x = 0u; x < width; x++) {
                for (auto c = 0u; c < stencil::NumChannels; c++) {
                    float16DataBuf[(y * width + x) * stencil::NumChannels +
                                   c] = half_float::half_cast<half_float::half>(
                            image.intensities[(y * width + x) * stencil::NumChannels + c]);
                }
            }
        }
    }

    auto inImageClBuf = cl::Buffer(context, CL_MEM_READ_WRITE,
                                   image.width * image.height * typeSize * stencil::NumChannels);
    auto outImageClBuf = cl::Buffer(context, CL_MEM_READ_WRITE,
                                    image.width * image.height * typeSize * stencil::NumChannels);

    auto in2Out = cl::Kernel(program, "gaussian_blur");
    in2Out.setArg(0, inImageClBuf);
    in2Out.setArg(1, outImageClBuf);
    in2Out.setArg(2, image.width);
    in2Out.setArg(3, image.height);

    auto out2in = cl::Kernel(program, "gaussian_blur");
    out2in.setArg(0, outImageClBuf);
    out2in.setArg(1, inImageClBuf);
    out2in.setArg(2, image.width);
    out2in.setArg(3, image.height);

    cl::vector<cl::Event> in2OutDone{1};
    cl::vector<cl::Event> out2InDone{1};

    if (dataType == "float") {
        cl::copy(queue, image.intensities.begin() , image.intensities.end() , inImageClBuf);
    } else {
        cl::copy(queue, float16DataBuf.begin() , float16DataBuf.end() , inImageClBuf);
    }
    cl::finish();

    std::cout << "Running Gaussian Blur " << numIters * 2 << " times";
    auto tic = std::chrono::high_resolution_clock::now();

//try{
    bool firstTime = true;
    auto timer = 0l;
    for (auto iter = 0u; iter < numIters; iter++) {
        auto err = queue.enqueueNDRangeKernel(
                in2Out,
                {1, 1}, // We only want to do the inner bit without the zero-padding
                {image.width - 2, image.height - 2},
                {TILE_WIDTH, TILE_HEIGHT},
                firstTime ? nullptr : &out2InDone,
                &in2OutDone[0]
        );
	if (err != 0) {
		std::cout << __LINE__ << err << std::endl;
	}
 	long time_start, time_end;
        in2OutDone[0].wait();
        in2OutDone[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
        in2OutDone[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
        timer += time_end - time_start;
        queue.enqueueNDRangeKernel(
                out2in,
                {1, 1}, // We only want to do the inner bit without the zero-padding
                {image.width - 2, image.height - 2},
                {TILE_WIDTH, TILE_HEIGHT},
		&in2OutDone,
                &out2InDone[0]
        );
	out2InDone[0].wait();
        out2InDone[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
        out2InDone[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
        timer += time_end - time_start;
        firstTime = false;
    }
    cl::finish();
    constexpr double NanoSeconds = 1e-9;
    std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << (double) timer * NanoSeconds << "s" <<
              std::endl;
//} catch (cl::Error &err) {
//	std::cout << err.err() << std::endl;
//}
    if (dataType == "half") {
        const auto height = image.height;
        const auto width = image.width;

        for (auto y = 0u; y < height; y++) {
            for (auto x = 0u; x < width; x++) {
                for (auto c = 0u; c < stencil::NumChannels; c++) {
                    image.intensities[(y * width + x) * stencil::NumChannels + c] = (float) (half_float::half) (
                            float16DataBuf[(y * width + x) * stencil::NumChannels + c]);
                }
            }
        }
    }

    if (dataType == "float") {
        cl::copy(queue, inImageClBuf, image.intensities.begin() , image.intensities.end() );
    } else {
        cl::copy(queue, inImageClBuf, float16DataBuf.begin() , float16DataBuf.end() );
    }
    cl::finish();


    auto cImg = stencil::toCharImage(stencil::stripPadding(image));

    if (!stencil::savePng(cImg, outputFilename)) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
