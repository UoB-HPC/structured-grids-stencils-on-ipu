

#include <cstdlib>

#include <iostream>
#include <algorithm>
#include <random>
#include <cxxopts.hpp>
#include <chrono>

#include "include/LbmParams.hpp"
#include "include/LatticeBoltzmannUtils.hpp"
#include "include/StructuredGridUtils.hpp"

#include <stdio.h>


#define HALO_OFFSET(r, c, speed) NumSpeeds * (((height + y - r) % height) * width + ((width + x + c) % width)) + speed
#define OFFSET(r, c, speed) HALO_OFFSET(r,c,speed)


namespace utils {
    const auto timedStep = [](const std::string description, auto f) -> double {
        std::cerr << std::setw(60) << description;
        auto tic = std::chrono::high_resolution_clock::now();
        f();
        auto toc = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast < std::chrono::duration < double >> (toc - tic).count();
        std::cerr << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;
        return diff;
    };
}

struct Cell {
    float speeds[9];
};


constexpr auto NumSpeeds = 9u;
enum Speed {
    Middle, East, North, West, South, NorthEast, NorthWest, SouthWest, SouthEast
};


inline auto lbmKernel(const Cell *__restrict nw, const Cell *__restrict n, const Cell *__restrict ne,
                      const Cell *__restrict w, const Cell *__restrict m, const Cell *__restrict e,
                      const Cell *__restrict sw, const Cell *__restrict s, const Cell *__restrict se,
                      const bool isObstacle, const bool isAccelerate, const float density,
                      const float omega, const float oneMinusOmega,
                      const float w1, const float w2) -> Cell {
    Cell result;

    // Streaming
    const float speed_nw = se->speeds[Speed::NorthWest];
    const float speed_n = s->speeds[Speed::North];
    const float speed_ne = sw->speeds[Speed::NorthEast];
    const float speed_w = e->speeds[Speed::West];
    const float speed_m = m->speeds[Speed::Middle];
    const float speed_e = w->speeds[Speed::East];
    const float speed_sw = ne->speeds[Speed::SouthWest];
    const float speed_s = n->speeds[Speed::South];
    const float speed_se = nw->speeds[Speed::SouthEast];

    if (isObstacle) {
        // rebound
        result.speeds[Speed::NorthWest] = speed_se;
        result.speeds[Speed::North] = speed_s;
        result.speeds[Speed::NorthEast] = speed_sw;
        result.speeds[Speed::West] = speed_e;
        result.speeds[Speed::Middle] = speed_m;
        result.speeds[Speed::East] = speed_w;
        result.speeds[Speed::SouthWest] = speed_ne;
        result.speeds[Speed::South] = speed_n;
        result.speeds[Speed::SouthEast] = speed_nw;
    } else {
        // collision (with acceleration folded in)
        const float local_density = speed_nw + speed_n + speed_ne +
                                    speed_w + speed_m + speed_e +
                                    speed_sw + speed_s + speed_se;
        /* compute x velocity component */
        const float u_x = (speed_e + speed_ne + speed_se - (speed_w + speed_ne + speed_sw)) / local_density;
        /* compute y velocity component */
        const float u_y = (speed_n + speed_nw + speed_ne - (speed_s + speed_sw + speed_se)) / local_density;
        const float u_sq = u_x * u_x + u_y * u_y;
        const float c_sq = 1.00f - u_sq * 1.50f;
        const float ld0 = 4.00f / 9.00f * local_density * omega;
        const float ld1 = local_density / 9.00f * omega;
        const float ld2 = local_density / 36.00f * omega;
        const float u_s = u_x + u_y;
        const float u_d = -u_x + u_y;

        const float speed_out_m = speed_m * oneMinusOmega + ld0 * c_sq;
        const float speed_out_e = speed_e * oneMinusOmega + ld1 * ((4.50f * u_x) * (2.00f / 3.00f + u_x) + c_sq);
        const float speed_out_n = speed_n * oneMinusOmega + ld1 * ((4.50f * u_y) * (2.00f / 3.00f + u_y) + c_sq);
        const float speed_out_w = speed_w * oneMinusOmega + ld1 * ((-4.50f * u_x) * (2.00f / 3.00f - u_x) + c_sq);
        const float speed_out_s = speed_s * oneMinusOmega + ld1 * ((-4.50f * u_y) * (2.00f / 3.00f - u_y) + c_sq);
        const float speed_out_ne = speed_ne * oneMinusOmega + ld2 * ((4.50f * u_s) * (2.00f / 3.00f + u_s) + c_sq);
        const float speed_out_nw = speed_nw * oneMinusOmega + ld2 * ((4.50f * u_d) * (2.00f / 3.00f + u_d) + c_sq);
        const float speed_out_sw = speed_sw * oneMinusOmega + ld2 * ((-4.50f * u_s) * (2.00f / 3.00f - u_s) + c_sq);
        const float speed_out_se = speed_se * oneMinusOmega + ld2 * ((-4.50f * u_d) * (2.00f / 3.00f - u_d) + c_sq);

        // Assign and fold in acceleration
        const float accelerateMask = isAccelerate ? 1.00f : 0.00f;

        result.speeds[Speed::NorthWest] = speed_out_nw - accelerateMask * w2;
        result.speeds[Speed::North] = speed_out_n;
        result.speeds[Speed::NorthEast] = speed_out_ne + accelerateMask * w2;
        result.speeds[Speed::West] = speed_out_w - accelerateMask * w1;;
        result.speeds[Speed::Middle] = speed_out_m;
        result.speeds[Speed::East] = speed_out_e + accelerateMask * w1;
        result.speeds[Speed::SouthWest] = speed_out_sw - accelerateMask * w2;
        result.speeds[Speed::South] = speed_out_s;
        result.speeds[Speed::SouthEast] = speed_out_se + accelerateMask * w2;
    }


    return result;
}


float timestep2(Cell *cells, Cell *tmp_cells, const bool *obstacles, const unsigned width, const unsigned height,
                const float density, const float accel, const float omega, const size_t num_free_cells) {
    /* compute weighting factors */
    const float w1 = density * accel / 9.0f;
    const float w2 = density * accel / 36.0f;
    const float one_minus_omega = 1.f - omega;

    float tot_u = 0.00f;          /* accumulated magnitudes of velocity for each cell */
    /* loop over the cells in the grid
    ** NB the collision step is called after
    ** the propagate step and so values of interest
    ** are in the scratch-space grid */
    for (int jj = 0; jj < height; jj++) {
        const int y_n = (jj + 1) % height;
        const int y_s = (jj == 0) ? (jj + height - 1) : (jj - 1);
        const float accel = (jj == height - 2) ? 1.00f : 0.00f;

        for (int ii = 0; ii < width; ii++) {
            const int x_e = (ii + 1) % width;
            const int x_w = (ii == 0) ? (ii + width - 1) : (ii - 1);
            const int is_obstacle = obstacles[ii + jj * width];

            const float speeds_0 = cells[ii + jj * width].speeds[0]; /* central cell, no movement */
            const float speeds_1 = cells[x_w + jj * width].speeds[1]; /* east */
            const float speeds_2 = cells[ii + y_s * width].speeds[2]; /* north */
            const float speeds_3 = cells[x_e + jj * width].speeds[3]; /* west */
            const float speeds_4 = cells[ii + y_n * width].speeds[4]; /* south */
            const float speeds_5 = cells[x_w + y_s * width].speeds[5]; /* north-east */
            const float speeds_6 = cells[x_e + y_s * width].speeds[6]; /* north-west */
            const float speeds_7 = cells[x_e + y_n * width].speeds[7]; /* south-west */
            const float speeds_8 = cells[x_w + y_n * width].speeds[8]; /* south-east */

            if (is_obstacle) {
                tmp_cells[ii + jj * width].speeds[0] = speeds_0;
                tmp_cells[ii + jj * width].speeds[1] = speeds_3;
                tmp_cells[ii + jj * width].speeds[2] = speeds_4;
                tmp_cells[ii + jj * width].speeds[3] = speeds_1;
                tmp_cells[ii + jj * width].speeds[4] = speeds_2;
                tmp_cells[ii + jj * width].speeds[5] = speeds_7;
                tmp_cells[ii + jj * width].speeds[6] = speeds_8;
                tmp_cells[ii + jj * width].speeds[7] = speeds_5;
                tmp_cells[ii + jj * width].speeds[8] = speeds_6;
            } else {

                /* compute local density total */
                const float local_density =
                        speeds_0 + speeds_1 + speeds_2 + speeds_3 + speeds_4 + speeds_5 + speeds_6 + speeds_7 +
                        speeds_8;
                /* compute x velocity component */
                const float u_x = (speeds_1 + speeds_5 + speeds_8 - (speeds_3 + speeds_6 + speeds_7)) / local_density;
                /* compute y velocity component */
                const float u_y = (speeds_2 + speeds_5 + speeds_6 - (speeds_4 + speeds_7 + speeds_8)) / local_density;

                /* velocity squared */
                const float u_sq = u_x * u_x + u_y * u_y;

                const float c_sq = 1.00f - u_sq * 1.50f;
                const float ld0 = 4.00f / 9.00f * local_density * omega;
                const float ld1 = local_density / 9.00f * omega;
                const float ld2 = local_density / 36.00f * omega;
                const float u_s = u_x + u_y;
                const float u_d = -u_x + u_y;

                const float speeds_out_0 = speeds_0 * one_minus_omega + ld0 * c_sq;
                const float speeds_out_1 =
                        speeds_1 * one_minus_omega + ld1 * ((4.50f * u_x) * (2.00f / 3.00f + u_x) + c_sq);
                const float speeds_out_2 =
                        speeds_2 * one_minus_omega + ld1 * ((4.50f * u_y) * (2.00f / 3.00f + u_y) + c_sq);
                const float speeds_out_3 =
                        speeds_3 * one_minus_omega + ld1 * ((-4.50f * u_x) * (2.00f / 3.00f - u_x) + c_sq);
                const float speeds_out_4 =
                        speeds_4 * one_minus_omega + ld1 * ((-4.50f * u_y) * (2.00f / 3.00f - u_y) + c_sq);
                const float speeds_out_5 =
                        speeds_5 * one_minus_omega + ld2 * ((4.50f * u_s) * (2.00f / 3.00f + u_s) + c_sq);
                const float speeds_out_6 =
                        speeds_6 * one_minus_omega + ld2 * ((4.50f * u_d) * (2.00f / 3.00f + u_d) + c_sq);
                const float speeds_out_7 =
                        speeds_7 * one_minus_omega + ld2 * ((-4.50f * u_s) * (2.00f / 3.00f - u_s) + c_sq);
                const float speeds_out_8 =
                        speeds_8 * one_minus_omega + ld2 * ((-4.50f * u_d) * (2.00f / 3.00f - u_d) + c_sq);

                tmp_cells[ii + jj * width].speeds[0] = speeds_out_0;
                tmp_cells[ii + jj * width].speeds[1] = speeds_out_1 + accel * w1;
                tmp_cells[ii + jj * width].speeds[2] = speeds_out_2;
                tmp_cells[ii + jj * width].speeds[3] = speeds_out_3 - accel * w1;
                tmp_cells[ii + jj * width].speeds[4] = speeds_out_4;
                tmp_cells[ii + jj * width].speeds[5] = speeds_out_5 + accel * w2;
                tmp_cells[ii + jj * width].speeds[6] = speeds_out_6 - accel * w2;
                tmp_cells[ii + jj * width].speeds[7] = speeds_out_7 - accel * w2;
                tmp_cells[ii + jj * width].speeds[8] = speeds_out_8 + accel * w2;
                tot_u += sqrtf(u_sq);

            }
        }
    }

    return tot_u / (float)num_free_cells;
}


inline auto normedLocalSpeed(const float nw, const float n, const float ne, const float w, const float m, const float e,
                             const float sw, const float s, const float se) -> float {
    // collision (with acceleration folded in)
    const float local_density = nw + n + ne +
                                w + m + e +
                                sw + s + se;
    /* compute x velocity component */
    const float u_x = (e + ne + se - (w + ne + sw)) / local_density;
    /* compute y velocity component */
    const float u_y = (n + nw + ne - (s + sw + se)) / local_density;
    return sqrtf(u_x * u_x + u_y * u_y);
}


auto doLbm(const float *in, float *out, const bool *obstacles, const unsigned rowToAccelerate,
           const unsigned width, const unsigned height, const float omega, const float accel,
           const float density) -> float {

    const float oneMinusOmega = 1 - omega;
    const float w1 = density * accel / 9.0f;
    const float w2 = density * accel / 36.0f;
    auto v = 0.f;
    for (auto y = 0u; y < height; y++) {
        const bool isAccelerate = y == rowToAccelerate;
        for (auto x = 0u; x < width; x++) {
            const Cell *nw = reinterpret_cast<const Cell *>(&(in[HALO_OFFSET(-1, -1, Speed::SouthEast)]));
            const Cell *n = reinterpret_cast<const Cell *>(&(in[HALO_OFFSET(-1, 0, 0)]));
            const Cell *ne = reinterpret_cast<const Cell *>(&(in[HALO_OFFSET(-1, 1, 0)]));
            const Cell *w = reinterpret_cast<const Cell *>(&(in[HALO_OFFSET(0, 1, 0)]));
            const Cell *m = reinterpret_cast<const Cell *>(&(in[HALO_OFFSET(0, 0, 0)]));
            const Cell *e = reinterpret_cast<const Cell *>(&(in[HALO_OFFSET(0, 1, 0)]));
            const Cell *sw = reinterpret_cast<const Cell *>(&(in[HALO_OFFSET(1, -1, 0)]));
            const Cell *s = reinterpret_cast<const Cell *>(&(in[HALO_OFFSET(1, 0, 0)]));
            const Cell *se = reinterpret_cast<const Cell *>(&(in[HALO_OFFSET(1, 1, 0)]));

            Cell *outPtr = reinterpret_cast<Cell *>(&out[OFFSET(0, 0, 0)]);
            const bool isObstacle = obstacles[x + y * width];

            const auto result = lbmKernel(nw, n, ne, w, m, e, sw, s, se, isObstacle,
                                          isAccelerate, density, omega, oneMinusOmega, w1, w2);

            v += normedLocalSpeed(result.speeds[Speed::NorthWest], result.speeds[Speed::North],
                                  result.speeds[Speed::NorthEast],
                                  result.speeds[Speed::West], result.speeds[Speed::Middle], result.speeds[Speed::East],
                                  result.speeds[Speed::SouthWest], result.speeds[Speed::South],
                                  result.speeds[Speed::SouthEast]);
            *outPtr = result;
        }
    }
    return v;

}


auto accelerate_flow(const lbm::Params &params, std::vector<float> &cells, const bool *obstacles) -> void {
    const auto accel = params.accel;
    const auto density = params.density;
    const auto width = params.nx;

    const float w1 = density * accel / 9.f;
    const float w2 = density * accel / 36.f;

    for (auto col = 0u; col < width; col++) {
        /* if the cell is not occupied and we don't send a negative density */
        const auto isObstacle = obstacles[(params.ny - 2) * width + col];
        const auto lidOffset = [&](size_t speed) -> size_t {
            return NumSpeeds * ((params.ny - 2) * width + col) + speed;
        };
        if (!isObstacle
            && (cells[lidOffset(Speed::West)] > w1)
            && (cells[lidOffset(Speed::NorthWest)] > w2)
            && (cells[lidOffset(Speed::SouthWest)] > w2)) {
            /* increase 'east-side' densities */
            cells[lidOffset(Speed::East)] += w1;
            cells[lidOffset(Speed::NorthEast)] += w2;
            cells[lidOffset(Speed::SouthEast)] += w2;
            /* decrease 'west-side' densities */
            cells[lidOffset(Speed::West)] -= w1;
            cells[lidOffset(Speed::SouthWest)] -= w2;
            cells[lidOffset(Speed::NorthWest)] -= w2;
        }
    }
}

auto timestep(const lbm::Params &params, std::vector<float> &cells, std::vector<float> &tmp_cells,
              const bool *obstacles, std::vector<float> &av_vels, size_t &idx, const size_t numFreeCells) {



//     Cell * a = reinterpret_cast< Cell *>(&(cells.data()[0]));
//     Cell * b = reinterpret_cast< Cell *>(&(tmp_cells.data()[0]));
//    av_vels[idx++] = timestep2(a, b, obstacles, params.nx, params.ny, params.density,
//                               params.accel, params.omega, numFreeCells);
    av_vels[idx++] = doLbm(cells.data(), tmp_cells.data(), obstacles, params.ny - 2, params.nx, params.ny, params.omega,
                           params.accel, params.density);
//    printf("==timestep: %d==\n", idx-1);
//    printf("av velocity: %.12E\n", av_vels[idx-1]);
//    printf("tot density: %.12E\n",   std::accumulate(tmp_cells.begin(), tmp_cells.end(), 0.0f));
//    for (auto jj = 0u; jj < params.ny; jj++) {
//        for (auto ii = 0u; ii < params.nx; ii++) {
//            for (auto kk = 0u; kk < 9; kk++) {
//                printf("%.12E ", tmp_cells[9 * (ii + jj * params.nx) + kk]);
//            }
//            printf("\n");
//        }
//        printf("\n");
//    }
    av_vels[idx++] = doLbm(tmp_cells.data(), cells.data(), obstacles, params.ny - 2, params.nx, params.ny, params.omega,
                           params.accel, params.density);
//    av_vels[idx++] = timestep2(b, a, obstacles, params.nx, params.ny, params.density,
//                               params.accel, params.omega, numFreeCells);
//    printf("==timestep: %d==\n", idx-1);
//    printf("av velocity: %.12E\n", av_vels[idx-1]);
//    printf("tot density: %.12E\n",    std::accumulate(cells.begin(), cells.end(), 0.0f));
//    for (auto jj = 0u; jj < params.ny; jj++) {
//        for (auto ii = 0u; ii < params.nx; ii++) {
//            for (auto kk = 0u; kk < 9; kk++) {
//                printf("%.12E ", cells[9 * (ii + jj * params.nx) + kk]);
//            }
//            printf("\n");
//        }
//        printf("\n");
//    }

}


auto main(int argc, char *argv[]) -> int {
    std::string obstaclesFileArg, paramsFileArg;
    cxxopts::Options options(argv[0], " - Runs a Lattice Boltzmann graph on the IPU or IPU Model");
    options.add_options()
            ("params", "filename of parameters file", cxxopts::value<std::string>(paramsFileArg))
            ("obstacles", "filename of obstacles file", cxxopts::value<std::string>(obstaclesFileArg));
    try {
        auto opts = options.parse(argc, argv);

        if (opts.count("params") + opts.count("obstacles") < 2) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    } catch (cxxopts::OptionParseException &) {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }


    auto params = lbm::Params::fromFile(paramsFileArg);
    if (!params.has_value()) {
        std::cerr << "Could not parse parameters file. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    auto obstacles = lbm::Obstacles::fromFile(params->nx, params->ny, obstaclesFileArg);
    if (!obstacles.has_value()) {
        std::cerr << "Could not parse obstacles file" << std::endl;
        return EXIT_FAILURE;
    }
    const auto numNonObstacleCells = [&]() -> long {
        long total = 0l;
        for (auto y = 0u; y < params->ny; y++) {
            for (auto x = 0u; x < params->nx; x++) {
                total += !obstacles->at(x, y);
            }
        }
        return total;
    }();

    auto cells = lbm::Cells(params->nx, params->ny);
    cells.initialise(*params);
    auto tmp_cells = std::vector<float>(params->nx * params->ny * lbm::NumSpeeds);
    auto av_vels = std::vector<float>(params->maxIters, 0.0f);

    std::cout << "Before!" << std::endl;
    std::cout << "Total Density: " << std::right << std::setw(12) << std::setprecision(12)
                                   << std::scientific<< cells.totalDensity() << std::endl;
    printf("density: %.12E\n", params->density);
    printf("accel: %.12E\n",   params->accel);
    printf("omega: %.12E\n",   params->omega);
    printf("reynolds_dim: %d\n",   params->reynolds_dim);
    printf("maxIters: %d\n",   params->maxIters);
    printf("nx: %d\n",   params->nx);
    printf("ny: %d\n",   params->ny);

    double total_compute_time = utils::timedStep("Compute \n", [&]() {
        accelerate_flow(*params, cells.data, obstacles->getData());
        size_t idx = 0;
        for (auto jj = 0u; jj < params->ny; jj++) {
            for (auto ii = 0u; ii < params->nx; ii++) {
                for (auto kk = 0u; kk < 9; kk++) {
                    printf("%.9f ", cells.data[9 * (ii + jj * params->nx) + kk]);
                }
                printf("\n");
            }
            printf("\n");
        }
        for (auto i = 0u; i < params->maxIters / 2; i++) {
            timestep(*params, cells.data, tmp_cells, obstacles->getData(), av_vels, idx, numNonObstacleCells);
        }
    });



    utils::timedStep("Writing output files ", [&]() {
//        for (auto i = 0u; i < av_vels.size(); i++) {
//            av_vels[i] = av_vels[i] / numNonObstacleCells;
//        }
        lbm::writeAverageVelocities("av_vels.dat", av_vels);
        lbm::writeResults("final_state.dat", *params, *obstacles, cells);
    });

    std::cout << "==done==" << std::endl;
    std::cout << "Total compute time was \t" << std::right << std::setw(12) << std::setprecision(5)
              << total_compute_time
              << "s" << std::endl;

    std::cout << "Reynolds number:  \t" << std::right << std::setw(12) << std::setprecision(12)
              << std::scientific
              << lbm::reynoldsNumber(*params, av_vels[params->maxIters - 1]) << std::endl;

    std::cout << "HOST total density: " << cells.totalDensity() << std::endl;


    return EXIT_SUCCESS;
}
