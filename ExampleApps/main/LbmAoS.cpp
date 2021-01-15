#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <iostream>
#include <poplar/Program.hpp>
#include <algorithm>
#include <random>
#include <cxxopts.hpp>
#include <popops/Zero.hpp>

#include "include/GraphcoreUtils.hpp"
#include "include/LbmParams.hpp"
#include "include/LatticeBoltzmannUtils.hpp"
#include "include/StructuredGridUtils.hpp"
#include <poplar/CycleCount.hpp>
#include <poplar/CSRFunctions.hpp>
#include <stdio.h>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace utils;

auto averageVelocity(Graph &graph, const lbm::Params &params, TensorMap &tensors,
                     const grids::GridPartitioning &workerLevelMappings) -> Program
{

    // As part of collision we already calculated a partialSum (float) and partialCount (unsigned) for each worker
    // which represents the summed normedVelocity and count of cells which are not masked by obstacles. Now we reduce them

    // Do multiple reductions in parallel
    std::vector<ComputeSet> reductionComputeSets;
    popops::reduceWithOutput(graph, tensors["perWorkerPartialSums"],
                             tensors["reducedSum"], {0}, {popops::Operation::ADD}, reductionComputeSets,
                             "reducedSums+=perWorkerPartialCounts[i]");

    /* Calculate the average and write it to the relevant place in the array. This happens on every tile,
     * because each tile owns a piece of cells, and only the owner of the piece with the index actually writes */

    const auto numIpus = graph.getTarget().getNumIPUs();
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / numIpus;

    auto ipuPartitioning = grids::partitionForIpus({params.maxIters, 1}, numIpus, params.maxIters / numIpus);
    assert(ipuPartitioning.has_value());
    auto avVelsTileMapping = grids::toTilePartitions(
        *ipuPartitioning,
        graph.getTarget().getNumIPUs(), 1, 10);

    ComputeSet appendResultCs = graph.addComputeSet("appendReducedSum");
    for (const auto &[target, slice] : avVelsTileMapping)
    {
        auto tile = target.virtualTile(numTilesPerIpu);
        auto appendReducedSumVertex = graph.addVertex(
            appendResultCs,
            "AppendReducedSum",
            {
                {"sumOfVelocities", tensors["reducedSum"]},
                {"indexToWrite", tensors["counter"]},
                {"finals", tensors["av_vel"].slice(
                                                slice.rows().from(),
                                                slice.rows().to(),
                                                0)
                               .flatten()},
            });
        graph.setInitialValue(appendReducedSumVertex["myStartIndex"], slice.rows().from());
        graph.setInitialValue(appendReducedSumVertex["myEndIndex"], slice.rows().to() - 1);
        graph.setTileMapping(tensors["av_vel"].slice(slice.rows().from(),
                                                     slice.rows().to(), 0),
                             tile);
        graph.setCycleEstimate(appendReducedSumVertex, 4);
        graph.setTileMapping(appendReducedSumVertex, tile);
    }

    ComputeSet incrementCs = graph.addComputeSet("increment");

    auto incrementVertex = graph.addVertex(incrementCs,
                                           "IncrementIndex", // Create a vertex of this
                                           {
                                               {"index", tensors["counter"]} // Connect input 'b' of the
                                           });
    graph.setCycleEstimate(incrementVertex, 13);
    graph.setTileMapping(incrementVertex, numTilesPerIpu - 1);

    Sequence seq;
    for (const auto &cs : reductionComputeSets)
    {
        seq.add(Execute(cs));
    }
    seq.add(Execute(appendResultCs));
    seq.add(Execute(incrementCs));
    return std::move(seq);
}
auto accelerate_flow(Graph &graph, const lbm::Params &params, TensorMap &tensors,
                     const unsigned numWorkersPerTile) -> Program
{

    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();

    ComputeSet accelerateCs = graph.addComputeSet("accelerate");

    auto cells = tensors["cells"];
    auto obstacles = tensors["obstacles"];

    auto cellsSecondRowFromTop = cells.slice(cells.dim(0) - 2, cells.dim(0) - 1, 0);
    auto obstaclesSecondRowFromTop = obstacles.slice(cells.dim(0) - 2, cells.dim(0) - 1, 0);

    const auto ipuLevelMapping = grids::partitionForIpus({1, params.nx}, graph.getTarget().getNumIPUs(),
                                                         params.nx);

    assert(ipuLevelMapping.has_value());
    auto tileGranularityMappings = grids::newTilePartitions(*ipuLevelMapping, numTilesPerIpu);
    auto workerGranularityMappings = grids::toWorkerPartitions(tileGranularityMappings);

    for (const auto &[target, slice] : workerGranularityMappings)
    {
        auto tile = target.virtualTile(numTilesPerIpu);
        auto numCellsForThisWorker = slice.width() * slice.height();
        auto v = graph.addVertex(accelerateCs,
                                 "FirstAccelerateVertex",
                                 {{"cellsVec", applySlice(cellsSecondRowFromTop, slice).flatten()},
                                  {"obstaclesVec", applySlice(obstaclesSecondRowFromTop,
                                                              slice)
                                                       .flatten()}});
        graph.setInitialValue(v["nx"], slice.width());
        graph.setInitialValue(v["density"], params.density);
        graph.setInitialValue(v["accel"], params.accel);
        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);
    }

    return Execute(accelerateCs);
}

auto two_timesteps(Graph &graph, const lbm::Params &params, TensorMap &tensors,
                   const grids::GridPartitioning &mappings, const unsigned numWorkersPerTile,
                   const int numNonObstacleCells) -> Program
{
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();

    ComputeSet in2outCs = graph.addComputeSet("in2out");
    ComputeSet out2inCs = graph.addComputeSet("out2in");
    auto fullSize = grids::Size2D(params.ny, params.nx);
    auto workerNum = 0;
    for (const auto &[target, slice] : mappings)
    {
        auto tile = target.virtualTile(numTilesPerIpu);
        auto numCellsForThisWorker = slice.width() * slice.height();
        const bool do_i_own_lid = slice.rows().from() <= params.ny - 2 && slice.rows().to() > params.ny - 2;
        const int which_of_my_rows_is_lid = do_i_own_lid ? params.ny - 2 - slice.rows().from() : -1;
        auto halos = grids::Halos::forSliceWithWraparound(slice, fullSize);
        auto stitched = utils::stitchHalos(utils::applySlice(tensors["cells"], *halos.topLeft),
                                           utils::applySlice(tensors["cells"], *halos.top),
                                           utils::applySlice(tensors["cells"], *halos.topRight),
                                           utils::applySlice(tensors["cells"], *halos.left),
                                           utils::applySlice(tensors["cells"], slice),
                                           utils::applySlice(tensors["cells"], *halos.right),
                                           utils::applySlice(tensors["cells"], *halos.bottomLeft),
                                           utils::applySlice(tensors["cells"], *halos.bottom),
                                           utils::applySlice(tensors["cells"], *halos.bottomRight));

        auto v = graph.addVertex(in2outCs,
                                 "LbmTimeStepVertex",
                                 {{"cells_oldVec", stitched.flatten()},
                                  {"cells_newVec", utils::applySlice(tensors["tmp_cells"], slice).flatten()},
                                  {"obstaclesVec", utils::applySlice(tensors["obstacles"], slice).flatten()},
                                  {"av_vel", tensors["perWorkerPartialSums"][workerNum]}});
        graph.setInitialValue(v["ny"], slice.height());
        graph.setInitialValue(v["nx"], slice.width());
        graph.setInitialValue(v["maxIters"], params.maxIters);
        graph.setInitialValue(v["omega"], params.omega);
        graph.setInitialValue(v["one_minus_omega"], 1.f - params.omega);
        graph.setInitialValue(v["density"], params.density);
        graph.setInitialValue(v["accel"], params.accel);
        graph.setInitialValue(v["isAccelerate"], do_i_own_lid);
        graph.setInitialValue(v["rowToAccelerate"], which_of_my_rows_is_lid);
        graph.setInitialValue(v["total_free_cells"], numNonObstacleCells);
        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);

        stitched = utils::stitchHalos(utils::applySlice(tensors["tmp_cells"], *halos.topLeft),
                                      utils::applySlice(tensors["tmp_cells"], *halos.top),
                                      utils::applySlice(tensors["tmp_cells"], *halos.topRight),
                                      utils::applySlice(tensors["tmp_cells"], *halos.left),
                                      utils::applySlice(tensors["tmp_cells"], slice),
                                      utils::applySlice(tensors["tmp_cells"], *halos.right),
                                      utils::applySlice(tensors["tmp_cells"], *halos.bottomLeft),
                                      utils::applySlice(tensors["tmp_cells"], *halos.bottom),
                                      utils::applySlice(tensors["tmp_cells"], *halos.bottomRight));

        v = graph.addVertex(out2inCs,
                            "LbmTimeStepVertex",
                            {{"cells_oldVec", stitched.flatten()},
                             {"cells_newVec", utils::applySlice(tensors["cells"], slice).flatten()},
                             {"obstaclesVec", utils::applySlice(tensors["obstacles"], slice).flatten()},
                             {"av_vel", tensors["perWorkerPartialSums"][workerNum]}});
        graph.setInitialValue(v["ny"], slice.height());
        graph.setInitialValue(v["nx"], slice.width());
        graph.setInitialValue(v["maxIters"], params.maxIters);
        graph.setInitialValue(v["omega"], params.omega);
        graph.setInitialValue(v["one_minus_omega"], 1.f - params.omega);
        graph.setInitialValue(v["density"], params.density);
        graph.setInitialValue(v["accel"], params.accel);
        graph.setInitialValue(v["isAccelerate"], do_i_own_lid);
        graph.setInitialValue(v["rowToAccelerate"], which_of_my_rows_is_lid);
        graph.setInitialValue(v["total_free_cells"], numNonObstacleCells);
        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);

        workerNum++;
    }

    const auto doAverage = averageVelocity(graph, params, tensors, mappings);

    return Sequence{Execute(in2outCs), doAverage, Execute(out2inCs), doAverage};
}

auto main(int argc, char *argv[]) -> int
{
    std::string obstaclesFileArg, paramsFileArg;
    unsigned numIpusArg = 1u;
    bool useModel = false;
    bool debug = false;
    cxxopts::Options options(argv[0], " - Runs a Lattice Boltzmann graph on the IPU or IPU Model");
    options.add_options()("d,debug", "Capture profiling")("m,use-ipu-model", "Use  IPU model instead of real device")("n,num-ipus", "number of ipus to use", cxxopts::value<unsigned>(numIpusArg)->default_value("1"))("params", "filename of parameters file", cxxopts::value<std::string>(paramsFileArg))("obstacles", "filename of obstacles file", cxxopts::value<std::string>(obstaclesFileArg));
    try
    {
        auto opts = options.parse(argc, argv);
        debug = opts["d"].as<bool>();
        useModel = opts["m"].as<bool>();
        if (opts.count("n") + opts.count("params") + opts.count("obstacles") < 3)
        {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    }
    catch (cxxopts::OptionParseException &)
    {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }
    if (debug)
    {
        std::cout << "Capturing profile information during this run." << std::endl;
    };

    auto params = lbm::Params::fromFile(paramsFileArg);
    if (!params.has_value())
    {
        std::cerr << "Could not parse parameters file. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    auto obstacles = lbm::Obstacles::fromFile(params->nx, params->ny, obstaclesFileArg);
    if (!obstacles.has_value())
    {
        std::cerr << "Could not parse obstacles file" << std::endl;
        return EXIT_FAILURE;
    }
    auto numNonObstacleCells = [&]() -> long {
        long total = 0l;
        for (auto y = 0u; y < obstacles->ny; y++)
        {
            for (auto x = 0u; x < obstacles->nx; x++)
            {
                total += !obstacles->at(x, y);
            }
        }
        return total;
    }();

    auto device = useModel ? utils::getIpuModel(numIpusArg) : utils::getIpuDevice(numIpusArg);
    if (!device.has_value())
    {
        return EXIT_FAILURE;
    }

    auto target = device->getTarget();
    auto numTilesPerIpu = target.getNumTiles() / target.getNumIPUs();
    auto numWorkersPerTile = target.getNumWorkerContexts();
    auto numIpus = target.getNumIPUs();

    const auto ipuLevelMapping = grids::partitionForIpus({params->ny, params->nx}, numIpus, 2000 * 1000);
    if (!ipuLevelMapping.has_value())
    {
        std::cerr << "Couldn't find a way to partition the input parameter space over the given number of IPUs"
                  << std::endl;
        return EXIT_FAILURE;
    }

    grids::serializeToJson(*ipuLevelMapping, "ipu-mapping.json");

    const auto tileGranularityMappings = grids::toTilePartitions(*ipuLevelMapping,
                                                                  numTilesPerIpu);

    const auto workerGranularityMappings = grids::toWorkerPartitions(
        tileGranularityMappings, numWorkersPerTile);

    grids::serializeToJson(workerGranularityMappings, "partitioning.json");

    auto tensors = std::map<std::string, Tensor>{};
    auto programs = std::vector<Program>{};

    Graph graph(target);

    timedStep("Building computational graph",
              [&]() {
                  popops::addCodelets(graph);

                  graph.addCodelets("codelets/D2Q9Codelets.cpp", CodeletFileType::Auto, debug ? "-g" : "-O3");

                  tensors["av_vel"] = graph.addVariable(FLOAT, {params->maxIters, 1},
                                                        "av_vel");
                  tensors["cells"] = graph.addVariable(FLOAT, {params->ny, params->nx, lbm::NumSpeeds},
                                                       "cells");
                  tensors["tmp_cells"] = graph.addVariable(FLOAT, {params->ny, params->nx, lbm::NumSpeeds},
                                                           "tmp_cells");
                  tensors["obstacles"] = graph.addVariable(BOOL, {params->ny, params->nx}, "obstacles");
                  tensors["perWorkerPartialSums"] = graph.addVariable(FLOAT,
                                                                      {numWorkersPerTile * numTilesPerIpu * numIpus},
                                                                      poplar::VariableMappingMethod::LINEAR,
                                                                      "perWorkerPartialSums");

                  tensors["reducedSum"] = graph.addVariable(FLOAT, {}, "reducedSum");
                  graph.setInitialValue(tensors["reducedSum"], 0.f);
                  graph.setTileMapping(tensors["reducedSum"], 0);

                  mapCellsToTiles(graph, tensors["cells"], tileGranularityMappings);
                  mapCellsToTiles(graph, tensors["tmp_cells"], tileGranularityMappings);
                  mapCellsToTiles(graph, tensors["obstacles"], tileGranularityMappings);

                  tensors["counter"] = graph.addVariable(UNSIGNED_INT, {}, "counter");
                  graph.setTileMapping(tensors["counter"], numTilesPerIpu - 1);
                  graph.setInitialValue(tensors["counter"], 0);

                  auto outStreamAveVelocities = graph.addDeviceToHostFIFO("<<av_vel", FLOAT, params->maxIters);
                  auto outStreamFinalCells = graph.addDeviceToHostFIFO("<<cells", FLOAT,
                                                                       lbm::NumSpeeds * params->nx *
                                                                           params->ny);
                  auto inStreamCells = graph.addHostToDeviceFIFO(">>cells", FLOAT,
                                                                 lbm::NumSpeeds * params->nx * params->ny);
                  auto inStreamObstacles = graph.addHostToDeviceFIFO(">>obstacles", BOOL,
                                                                     params->nx * params->ny);

                  auto streamBackToHostProg = Sequence(
                      Copy(tensors["cells"], outStreamFinalCells),
                      Copy(tensors["av_vel"], outStreamAveVelocities));

                  auto prog = Sequence();
                  poplar::setFloatingPointBehaviour(graph, prog, {true, true, true, false, true},
                                                    "no stochastic rounding");
                  prog.add(accelerate_flow(graph, *params, tensors, numWorkersPerTile));
                  prog.add(
                      Repeat(params->maxIters / 2,
                             two_timesteps(graph, *params, tensors, workerGranularityMappings,
                                           numWorkersPerTile, numNonObstacleCells)));

                  auto copyCellsAndObstaclesToDevice = Sequence();
                  popops::zero(graph, tensors["av_vel"], copyCellsAndObstaclesToDevice, "av_vels=0");
                  popops::zero(graph, tensors["cells"], copyCellsAndObstaclesToDevice, "cells=0");
                  popops::zero(graph, tensors["tmp_cells"], copyCellsAndObstaclesToDevice, "tmp_cells=0");
                  copyCellsAndObstaclesToDevice.add(Copy(inStreamCells, tensors["cells"]));
                  copyCellsAndObstaclesToDevice.add(Copy(inStreamObstacles, tensors["obstacles"]));

                  auto timing = poplar::cycleCount(graph,
                                                   prog,
                                                   0, "timer");

                  graph.createHostRead("readTimer", timing, true);
                  programs.push_back(copyCellsAndObstaclesToDevice);
                  programs.push_back(prog);
                  programs.push_back(streamBackToHostProg);

                  if (auto dumpGraphVisualisations =
                          std::getenv("DUMP_GRAPH_VIZ") != nullptr;
                      dumpGraphVisualisations)
                  {
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
              });

    timedStep("Serializing graph for analysis", [&]() -> void {
        serializeGraph(graph);
    });

    double total_compute_time = 0.0;
    auto cells = lbm::Cells(params->nx, params->ny);
    cells.initialise(*params);

    auto av_vels = std::vector<float>(params->maxIters, 0.0f);

    auto engine = Engine(graph, programs,
                         debug ? utils::POPLAR_ENGINE_OPTIONS_DEBUG : utils::POPLAR_ENGINE_OPTIONS_NODEBUG);

    engine.connectStream("<<av_vel", av_vels.data());
    engine.connectStream("<<cells", cells.data.data());
    engine.connectStream(">>cells", cells.data.data());
    engine.connectStream(">>obstacles", obstacles->getData());

    utils::timedStep("Loading graph to device", [&]() {
        engine.load(*device);
    });

    utils::timedStep("Running copy to device step", [&]() {
        engine.run(0);
    });

    total_compute_time += utils::timedStep("Running LBM", [&]() {
        engine.run(1);
    });

    utils::timedStep("Running copy to host step", [&]() {
        engine.run(2);
    });

    utils::timedStep("Writing output files ", [&]() {
        lbm::writeAverageVelocities("av_vels.dat", av_vels);
        lbm::writeResults("final_state.dat", *params, *obstacles, cells);
    });

    if (debug)
    {
        utils::timedStep("Capturing profiling info", [&]() {
            utils::captureProfileInfo(engine);
        });

        engine.printProfileSummary(std::cout,
                                   OptionFlags{{"showExecutionSteps", "false"}});
    }

    std::cout << "==done==" << std::endl;
    std::cout << "Total compute time was \t" << std::right << std::setw(12) << std::setprecision(5)
              << total_compute_time
              << "s" << std::endl;

    std::cout << "Reynolds number:  \t" << std::right << std::setw(12) << std::setprecision(12)
              << std::scientific
              << lbm::reynoldsNumber(*params, av_vels[params->maxIters - 1]) << std::endl;

    std::cout << "HOST total density: " << cells.totalDensity() << std::endl;

    cout << "Now doing 5 runs and averaging IPU-reported timing:" << std::endl;
    unsigned long ipuTimer;
    double clockCycles = 0.;
    for (auto run = 0u; run < 5u; run++)
    {
        engine.run(2);
        engine.readTensor("readTimer", &ipuTimer);
        clockCycles += ipuTimer;
    }
    double clockFreq = device->getTarget().getTileClockFrequency();
    std::cout << "IPU reports " << std::fixed << clockFreq * 1e-6 << "MHz clock frequency" << std::endl;
    std::cout << "Average IPU timing for program is: " << std::fixed << std::setprecision(5) << std::setw(12)
              << clockCycles / 5.0 / clockFreq << "s" << std::endl;

    std::cout << "==done==" << std::endl;

    return EXIT_SUCCESS;
}
