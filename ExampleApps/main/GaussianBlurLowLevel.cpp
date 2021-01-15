
#include <cstdlib>
#include <cxxopts.hpp>
#include <lodepng.h>
#include "include/StructuredGridUtils.hpp"
#include <chrono>
#include "include/ImageUtils.hpp"
#include "include/GraphcoreUtils.hpp"
#include <poplar/IPUModel.hpp>
#include <popops/codelets.hpp>
#include <popops/Zero.hpp>
#include <poplar/CSRFunctions.hpp>
#include <iostream>
#include <poplar/Program.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <popfloat/experimental/CastToHalf.hpp> // for halfToSingle and singleToHalf
#include <poplar/CycleCount.hpp>
#include <sstream>

using namespace stencil;

const auto vertexName(const grids::Slice2D _slice, const std::string dataType) -> std::string {
    if (dataType == "float2")
        return "GaussianBlurCodeletFloat2";
    else if (dataType == "half4")
        return "GaussianBlurCodeletHalf4";
    return std::string("GaussianBlurCodelet<").append(dataType).append(">");
}


int main(int argc, char *argv[]) {
    std::string inputFilename, outputFilename;
    unsigned numIters = 1u;
    unsigned numIpus = 1u;
    unsigned minRowsPerTile = 6u;
    unsigned minColsPerTile = 6u;
    bool compileOnly = false;
    bool debug = false;
    bool useIpuModel = false;
    std::string dataType = "float";


    cxxopts::Options options(argv[0],
                             " - Runs a 3x3 Gaussian Blur stencil over an input image a number of times using the IPU");
    options.add_options()
            ("n,num-iters", "Number of iterations", cxxopts::value<unsigned>(numIters)->default_value("1"))
            ("i,image", "filename input image (must be a 4-channel PNG image)",
             cxxopts::value<std::string>(inputFilename))
            ("num-ipus", "Number of IPUs to target (1,2,4,8 or 16)", cxxopts::value<unsigned>(numIpus))
            ("min-rows-per-tile", "Min rows per tile (default 6)",
             cxxopts::value<unsigned>(minRowsPerTile)->default_value(std::to_string(grids::DefaultMinRowsPerTile)))
            ("min-cols-per-tile", "Min cols per tile (default 6)",
             cxxopts::value<unsigned>(minColsPerTile)->default_value(std::to_string(grids::DefaultMinColsPerTile)))
            ("data-type", "Data type (float, float2, half, half4)",
             cxxopts::value<string>(dataType)->default_value("float"))
            ("o,output", "filename output (blurred image)", cxxopts::value<std::string>(outputFilename))
            ("d,debug", "Run in debug mode (capture profiling information)")
            ("compile-only", "Only compile the graph and write to stencil_<width>x<height>.exe, don't run")
            ("m,ipu-model", "Run on IPU model (emulator) instead of real device");

    try {
        auto opts = options.parse(argc, argv);
        debug = opts["debug"].as<bool>();
        compileOnly = opts["compile-only"].as<bool>();
        useIpuModel = opts["ipu-model"].as<bool>();
        if (opts.count("n") + opts.count("i") + opts.count("num-ipus") + opts.count("o") < 4) {
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

    auto fImage = toFloatImage(*maybeImg);
    auto img = fImage.intensities.data();
    void *dataBuf = img; // the float case
    uint16_t *float16DataBuf = nullptr;
    if (dataType.rfind("half", 0) == 0) {
        const auto height = fImage.height;
        const auto width = fImage.width;
        float16DataBuf = new uint16_t[width * height * NumChannels];

#pragma omp parallel for  default(none) shared(img, float16DataBuf)  schedule(static, 4) collapse(3)
        for (auto y = 0u; y < height; y++) {
            for (auto x = 0u; x < width; x++) {
                for (auto c = 0u; c < NumChannels; c++) {
                    float16DataBuf[(y * width + x) * NumChannels + c] = popfloat::experimental::singleToHalf(
                            img[(y * width + x) * NumChannels + c]);
                }
            }
        }
        dataBuf = float16DataBuf;
    }

    auto device = useIpuModel ? utils::getIpuModel(numIpus) : utils::getIpuDevice(numIpus);
    if (!device.has_value()) {
        return EXIT_FAILURE;
    }

    std::cout << "Building graph";
    auto tic = std::chrono::high_resolution_clock::now();
    auto graph = poplar::Graph(*device);
    auto poplarType = (dataType.rfind("half", 0)) == 0 ? HALF : FLOAT;
    graph.addCodelets("codelets/GaussianBlurCodelets.cpp", CodeletFileType::Auto, "-O3");
    graph.addCodelets("codelets/GaussianBlurCodeletsVectorised.cpp", CodeletFileType::Auto, "-O3");

    const auto inImg = graph.addHostToDeviceFIFO(">>img", poplarType,
                                                 NumChannels * fImage.height * fImage.width);
    const auto outImg = graph.addDeviceToHostFIFO("<<img", poplarType,
                                                  NumChannels * fImage.height * fImage.width);

    auto imgTensor = graph.addVariable(poplarType, {fImage.height, fImage.width, NumChannels}, "img");
    auto tmpImgTensor = graph.addVariable(poplarType, {fImage.height, fImage.width, NumChannels}, "tmpImg");

    auto ipuLevelMappings = grids::partitionForIpus({fImage.height, fImage.width}, numIpus, 2000 * 1400);
    if (!ipuLevelMappings.has_value()) {
        std::cerr << "Couldn't fit the problem on the " << numIpus << " ipus." << std::endl;
        return EXIT_FAILURE;
    }
    auto tileLevelMappings = grids::toTilePartitions(*ipuLevelMappings, graph.getTarget().getTilesPerIPU(),
                                                     minRowsPerTile,
                                                     minColsPerTile);
    auto numWorkersPerTile = device->getTarget().getNumWorkerContexts();
    auto workerLevelMappings = grids::toWorkerPartitions(tileLevelMappings, numWorkersPerTile);
    grids::serializeToJson(workerLevelMappings, "partitions.json");
    for (const auto &[target, slice]: tileLevelMappings) {
        graph.setTileMapping(utils::applySlice(imgTensor, slice), target.virtualTile());
        graph.setTileMapping(utils::applySlice(tmpImgTensor, slice), target.virtualTile());
    }

    auto inToOut = graph.addComputeSet("inToOut");
    auto outToIn = graph.addComputeSet("outToIn");
    auto zeros = std::vector<float>(std::max(fImage.width, fImage.height), 0.f);
    for (const auto &[target, slice]: workerLevelMappings) {
        // Halos top-left = (0,0) and no wraparound
        const auto halos = grids::Halos::forSliceTopIs0NoWrap(slice, {fImage.height, fImage.width});

        const auto maybeZerosVector = std::optional < Tensor > {};
        const auto maybeZeroScalar = std::optional < Tensor > {};

        const auto _target = target;
        const auto applyOrZero = [&graph, &_target, &zeros, poplarType](const std::optional <grids::Slice2D> maybeSlice,
                                                                        Tensor &tensor,
                                                                        const unsigned borderRows = 1u,
                                                                        const unsigned borderCols = 1u) -> Tensor {
            if (maybeSlice.has_value()) {
                return utils::applySlice(tensor, *maybeSlice);
            } else {
                const auto zerosVector = graph.addConstant(poplarType, {borderRows, borderCols, 4}, zeros.data(),
                                                           "{0...}");
                graph.setTileMapping(zerosVector, _target.virtualTile());
                return zerosVector;
            }
        };

        auto n = applyOrZero(halos.top, imgTensor, 1, slice.width());
        auto s = applyOrZero(halos.bottom, imgTensor, 1, slice.width());
        auto e = applyOrZero(halos.right, imgTensor, slice.height(), 1);
        auto w = applyOrZero(halos.left, imgTensor, slice.height(), 1);
        auto nw = applyOrZero(halos.topLeft, imgTensor);
        auto ne = applyOrZero(halos.topRight, imgTensor);
        auto sw = applyOrZero(halos.bottomLeft, imgTensor);
        auto se = applyOrZero(halos.bottomRight, imgTensor);
        auto m = utils::applySlice(imgTensor, slice);

        const auto inWithHalos = utils::stitchHalos(nw, n, ne, w, m, e, sw, s, se);

        auto v = graph.addVertex(inToOut,
                                 vertexName(slice, dataType),
                                 {
                                         {"in",  inWithHalos.flatten()},
                                         {"out", utils::applySlice(tmpImgTensor, slice).flatten()},
                                 }
        );
        graph.setInitialValue(v["width"], slice.width());
        graph.setInitialValue(v["height"], slice.height());
        graph.setCycleEstimate(v, slice.width() * slice.height() * 11);
        graph.setTileMapping(v, target.virtualTile());
        n = applyOrZero(halos.top, tmpImgTensor, 1, slice.width());
        s = applyOrZero(halos.bottom, tmpImgTensor, 1, slice.width());
        e = applyOrZero(halos.right, tmpImgTensor, slice.height(), 1);
        w = applyOrZero(halos.left, tmpImgTensor, slice.height(), 1);
        nw = applyOrZero(halos.topLeft, tmpImgTensor);
        ne = applyOrZero(halos.topRight, tmpImgTensor);
        sw = applyOrZero(halos.bottomLeft, tmpImgTensor);
        se = applyOrZero(halos.bottomRight, tmpImgTensor);
        m = utils::applySlice(tmpImgTensor, slice);
        const auto outWithHalos = utils::stitchHalos(nw, n, ne, w, m, e, sw, s, se);

        v = graph.addVertex(outToIn,
                            vertexName(slice, dataType),
                            {
                                    {"out", utils::applySlice(imgTensor, slice).flatten()},
                                    {"in",  outWithHalos.flatten()}
                            }
        );
        graph.setInitialValue(v["width"], slice.width());
        graph.setInitialValue(v["height"], slice.height());
        graph.setCycleEstimate(v, slice.width() * slice.height() * 9);
        graph.setTileMapping(v, target.virtualTile());
    }
    Sequence stencilProgram;
    poplar::setFloatingPointBehaviour(graph, stencilProgram, {true, true, true, false, true}, "no stochastic rounding");
    stencilProgram.add(Execute(inToOut));
    stencilProgram.add(Execute(outToIn));

    Sequence timedStencilProgram = Repeat(numIters, stencilProgram);

    auto dummyTime = std::array<unsigned,2>{0};
    const auto dummyTimeForModel = [&]() -> Tensor  {
        auto tmp = graph.addConstant(UNSIGNED_INT, {1, 2}, dummyTime.data(), "{0...}");
        graph.setTileMapping(tmp, 1215);
        return tmp;
    };
    auto timing = useIpuModel ? dummyTimeForModel() :
                  poplar::cycleCount(graph,
                                     timedStencilProgram,
                                     0, "timer");


    graph.createHostRead("readTimer", timing, true);
    auto copyToDevice = Copy(inImg, imgTensor);
    auto copyBackToHost = Copy(imgTensor, outImg);

    const auto programs = std::vector < Program > {copyToDevice,
                                                   timedStencilProgram,
                                                   copyBackToHost};


    auto toc = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast < std::chrono::duration < double >> (toc - tic).count();
    std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
              std::endl;

    if (debug) {
        utils::serializeGraph(graph);
    }
    if (auto dumpGraphVisualisations =
                std::getenv("DUMP_GRAPH_VIZ") != nullptr;  dumpGraphVisualisations) {
        ofstream vertexGraph;
        vertexGraph.open("vertexgraph.dot");
        graph.outputVertexGraph(vertexGraph,
                                programs);
        vertexGraph.close();

        ofstream computeGraph;
        computeGraph.open("computegraph.dot");
        graph.outputComputeGraph(computeGraph,
                                 programs);
        computeGraph.close();
    }
    std::cout << "Compiling graph";
    tic = std::chrono::high_resolution_clock::now();
    if (compileOnly) {
        auto exe = poplar::compileGraph(graph, programs, debug ? utils::POPLAR_ENGINE_OPTIONS_DEBUG
                                                               : utils::POPLAR_ENGINE_OPTIONS_NODEBUG);
        toc = std::chrono::high_resolution_clock::now();
        diff = std::chrono::duration_cast < std::chrono::duration < double >> (toc - tic).count();
        std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
                  std::endl;

        stringstream filename;
        filename << "stencil_" << maybeImg->width << "x" << maybeImg->height << ".exe";
        ofstream exe_file;
        exe_file.open(filename.str());
        exe.serialize(exe_file);
        exe_file.close();

        return EXIT_SUCCESS;
    } else {
        auto engine = Engine(graph, programs,
                             debug ? utils::POPLAR_ENGINE_OPTIONS_DEBUG : utils::POPLAR_ENGINE_OPTIONS_NODEBUG);

        toc = std::chrono::high_resolution_clock::now();
        diff = std::chrono::duration_cast < std::chrono::duration < double >> (toc - tic).count();
        std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
                  std::endl;


        engine.connectStream(">>img", dataBuf);
        engine.connectStream("<<img", dataBuf);

        engine.load(*device);

        utils::timedStep("Copying to device", [&]() -> void {
            engine.run(0);
        });

        utils::timedStep("Running stencil iterations", [&]() -> void {
            engine.run(1);
        });

        utils::timedStep("Copying to host", [&]() -> void {
            engine.run(2);
        });

        if (debug) {
            utils::captureProfileInfo(engine);

            engine.printProfileSummary(std::cout,
                                       OptionFlags{{"showExecutionSteps", "false"}});
        }

        if (dataType.rfind("half", 0) == 0) {
            const auto height = fImage.height;
            const auto width = fImage.width;
#pragma omp parallel for  default(none) shared(img, float16DataBuf)  schedule(static, 4) collapse(3)
            for (auto y = 0u; y < height; y++) {
                for (auto x = 0u; x < width; x++) {
                    for (auto c = 0u; c < NumChannels; c++) {
                        img[(y * width + x) * NumChannels + c] = popfloat::experimental::halfToSingle(
                                float16DataBuf[(y * width + x) * NumChannels + c]);
                    }
                }
            }

            delete float16DataBuf;
        }

        auto cImg = toCharImage(fImage);


        if (!savePng(cImg, outputFilename)) {
            return EXIT_FAILURE;
        }
        cout << "Now doing 5 runs and averaging IPU-reported timing:" << std::endl;
        unsigned long ipuTimer;
        double clockCycles = 0.;
        for (auto run = 0u; run < 5u; run++) {
            engine.run(1);
            engine.readTensor("readTimer", &ipuTimer);
            clockCycles += ipuTimer;
        }
        double clockFreq = device->getTarget().getTileClockFrequency();
        std::cout << "IPU reports " << clockFreq * 1e-6 << "MHz clock frequency" << std::endl;
        std::cout << "Average IPU timing for program is: " << std::setprecision(5)
                  << clockCycles / 5.0 / clockFreq << "s" << std::endl;
    }

    return EXIT_SUCCESS;
}
