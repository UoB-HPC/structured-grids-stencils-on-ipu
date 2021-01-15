//
// Created by Thorben Louw on 25/06/2020.
//

#ifndef LBM_GRAPHCORE_LATTICEBOLTZMANN_H
#define LBM_GRAPHCORE_LATTICEBOLTZMANN_H

#include <cmath>
#include <numeric>
#include <iostream>
#include <iomanip>
#include "LbmParams.hpp"


namespace lbm {
    constexpr auto NumSpeeds = 9u;

#define AT_SPEED(s) [(ii + jj * nx) * 9 + s]

    enum SpeedIndexes {
        Middle, East, North, West, South, NorthEast, NorthWest, SouthWest, SouthEast
    };


    class CellsSoA {

    public:
        const size_t nx;
        const size_t ny;
        std::vector<float> middleSpeeds;
        std::vector<float> northSpeeds;
        std::vector<float> southSpeeds;
        std::vector<float> eastSpeeds;
        std::vector<float> westSpeeds;
        std::vector<float> northEastSpeeds;
        std::vector<float> northWestSpeeds;
        std::vector<float> southEastSpeeds;
        std::vector<float> southWestSpeeds;

        explicit CellsSoA(size_t nx, size_t ny) : nx(nx), ny(ny) {
            middleSpeeds = std::vector<float>(nx * ny, 0.f);
            northSpeeds = std::vector<float>(nx * ny, 0.f);
            southSpeeds = std::vector<float>(nx * ny, 0.f);
            eastSpeeds = std::vector<float>(nx * ny, 0.f);
            westSpeeds = std::vector<float>(nx * ny, 0.f);
            northEastSpeeds = std::vector<float>(nx * ny, 0.f);
            northWestSpeeds = std::vector<float>(nx * ny, 0.f);
            southEastSpeeds = std::vector<float>(nx * ny, 0.f);
            southWestSpeeds = std::vector<float>(nx * ny, 0.f);
        }

        auto initialise(const Params &params) -> void {
            float w0 = params.density * 4.f / 9.f;
            float w1 = params.density / 9.f;
            float w2 = params.density / 36.f;

            auto nx = params.nx;
            auto ny = params.ny;
            for (auto jj = 0u; jj < ny; jj++) {
                for (auto ii = 0u; ii < nx; ii++) {
                    middleSpeeds[jj * nx + ii] = w0;
                    northSpeeds[jj * nx + ii] = w1;
                    southSpeeds[jj * nx + ii] = w1;
                    eastSpeeds[jj * nx + ii] = w1;
                    westSpeeds[jj * nx + ii] = w1;
                    northEastSpeeds[jj * nx + ii] = w2;
                    northWestSpeeds[jj * nx + ii] = w2;
                    southEastSpeeds[jj * nx + ii] = w2;
                    southWestSpeeds[jj * nx + ii] = w2;
                }
            }
        }


        auto averageVelocity(const Params &params, const Obstacles &obstacles) const -> float {
            int tot_cells = 0;  /* no. of cells used in calculation */
            float tot_u;          /* accumulated magnitudes of velocity for each cell */

            /* initialise */
            tot_u = 0.f;

            /* loop over all non-blocked cells */
            for (auto jj = 0u; jj < params.ny; jj++) {
                for (auto ii = 0u; ii < params.nx; ii++) {
                    int mask = 1 - obstacles.at(ii, jj);
                    const auto idx = jj * nx + ii;

                    float local_density = middleSpeeds[idx] + northSpeeds[idx] + southSpeeds[idx] +
                                          eastSpeeds[idx] + westSpeeds[idx] + northWestSpeeds[idx] +
                                          northEastSpeeds[idx] + southWestSpeeds[idx] + southEastSpeeds[idx];

                    /* x-component of velocity */
                    float u_x = (float) mask * (eastSpeeds[idx] + northEastSpeeds[idx] + southEastSpeeds[idx] -
                                                (westSpeeds[idx] + northWestSpeeds[idx] + southWestSpeeds[idx])) /
                                local_density;
                    /* compute y velocity component */
                    float u_y = (float) mask * (northSpeeds[idx] + northEastSpeeds[idx] + northWestSpeeds[idx] -
                                                (southEastSpeeds[idx] + southWestSpeeds[idx] + southSpeeds[idx])) /
                                local_density;
                    /* accumulate the norm of x- and y- velocity components */
                    tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
                    /* increase counter of inspected cells */
                    tot_cells += mask;

                }
            }

            return tot_u / (float) tot_cells;
        }

        auto totalDensity() const -> float {
            return std::accumulate(middleSpeeds.begin(), middleSpeeds.end(), 0.0f) +
                   std::accumulate(northSpeeds.begin(), northSpeeds.end(), 0.0f) +
                   std::accumulate(southSpeeds.begin(), southSpeeds.end(), 0.0f) +
                   std::accumulate(westSpeeds.begin(), westSpeeds.end(), 0.0f) +
                   std::accumulate(eastSpeeds.begin(), eastSpeeds.end(), 0.0f) +
                   std::accumulate(northWestSpeeds.begin(), northWestSpeeds.end(), 0.0f) +
                   std::accumulate(northEastSpeeds.begin(), northEastSpeeds.end(), 0.0f) +
                   std::accumulate(southWestSpeeds.begin(), southWestSpeeds.end(), 0.0f) +
                   std::accumulate(southEastSpeeds.begin(), southEastSpeeds.end(), 0.0f);
        }
    };


    class Cells {

    public:
        const size_t nx;
        const size_t ny;
        std::vector<float> data;

        explicit Cells(size_t nx, size_t ny) :
                nx(nx), ny(ny) {
            data = std::vector<float>(nx * ny * NumSpeeds, 0.f);
        }

        auto initialise(const Params &params) -> void {
            float w0 = params.density * 4.f / 9.f;
            float w1 = params.density / 9.f;
            float w2 = params.density / 36.f;

            auto nx = params.nx;
            auto ny = params.ny;
            for (auto jj = 0u; jj < ny; jj++) {
                for (auto ii = 0u; ii < nx; ii++) {
                    data AT_SPEED(0) = w0;
                    data AT_SPEED(1) = w1;
                    data AT_SPEED(2) = w1;
                    data AT_SPEED(3) = w1;
                    data AT_SPEED(4) = w1;
                    data AT_SPEED(5) = w2;
                    data AT_SPEED(6) = w2;
                    data AT_SPEED(7) = w2;
                    data AT_SPEED(8) = w2;
                }
            }
        }


        auto averageVelocity(const Params &params, const Obstacles &obstacles) const -> float {
            int tot_cells = 0;  /* no. of cells used in calculation */
            float tot_u;          /* accumulated magnitudes of velocity for each cell */

            /* initialise */
            tot_u = 0.f;

            /* loop over all non-blocked cells */
            for (auto jj = 0u; jj < params.ny; jj++) {
                for (auto ii = 0u; ii < params.nx; ii++) {
                    int mask = 1 - obstacles.at(ii, jj);

                    float local_density = 0.f;
                    for (unsigned s = 0u; s < NumSpeeds; s++) {
                        local_density += data AT_SPEED(s);
                    }

                    /* x-component of velocity */
                    float u_x = (float) mask * (data AT_SPEED(1) + data AT_SPEED(5) + data AT_SPEED(8) -
                                                (data AT_SPEED(3) + data AT_SPEED(6) + data AT_SPEED(7))) /
                                local_density;
                    /* compute y velocity component */
                    float u_y = (float) mask * ((data AT_SPEED(2) + data AT_SPEED(5) + data AT_SPEED(6)) -
                                                (data AT_SPEED(4) + data AT_SPEED(7) + data AT_SPEED(8))) /
                                local_density;
                    /* accumulate the norm of x- and y- velocity components */
                    tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
                    /* increase counter of inspected cells */
                    tot_cells += mask;

                }
            }

            return tot_u / (float) tot_cells;
        }

        auto totalDensity() const -> float {
            return std::accumulate(data.begin(), data.end(), 0.0f);
        }
    };


    auto reynoldsNumber(const Params &params, float average_velocity) -> float {
        const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
        return average_velocity * params.reynolds_dim / viscosity;
    }


    auto writeAverageVelocities(const std::string &filename, const std::vector<float> &av_vels) -> bool {
        std::ofstream file;
        file.open(filename, std::ios::out);
        if (file.is_open()) {
            for (auto i = 0ul; i < av_vels.size(); i++) {
                file << i << ":\t" << std::scientific << std::setprecision(12) << av_vels[i] << std::endl;
            }
            file.close();
            return true;
        }
        return false;
    }

    auto writeResults(const std::string &filename,
                      const Params &params,
                      const Obstacles &obstacles,
                      const Cells &cells) -> bool {
        const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
        float local_density;         /* per grid cell sum of densities */
        float pressure;              /* fluid pressure in grid cell */
        float u_x;                   /* x-component of velocity in grid cell */
        float u_y;                   /* y-component of velocity in grid cell */
        float u;                     /* norm--root of summed squares--of u_x and u_y */

        std::ofstream file;
        file.open(filename, std::ios::out);
        if (file.is_open()) {

            const auto nx = params.nx;
            const auto ny = params.ny;
            for (auto jj = 0u; jj < ny; jj++) {
                for (auto ii = 0u; ii < nx; ii++) {
                    /* an occupied cell */
                    if (obstacles.at(ii, jj)) {
                        u_x = u_y = u = 0.f;
                        pressure = params.density * c_sq;
                    }
                        /* no obstacle */
                    else {
                        local_density = 0.f;

                        for (auto kk = 0u; kk < NumSpeeds; kk++) {
                            local_density += cells.data AT_SPEED(kk);
                        }

                        /* compute x velocity component */
                        u_x = (cells.data AT_SPEED(1) + cells.data AT_SPEED(5) + cells.data AT_SPEED(8)
                               - (cells.data AT_SPEED(3) + cells.data AT_SPEED(6) + cells.data AT_SPEED(7)))
                              / local_density;
                        /* compute y velocity component */
                        u_y = (cells.data AT_SPEED(2) + cells.data AT_SPEED(5) + cells.data AT_SPEED(6)
                               - (cells.data AT_SPEED(4) + cells.data AT_SPEED(7) + cells.data AT_SPEED(8)))
                              / local_density;


                        /* compute norm of velocity */
                        u = sqrtf((u_x * u_x) + (u_y * u_y));
                        /* compute pressure */
                        pressure = local_density * c_sq;
                    }

                    /* write to file */
                    file << ii << " " << jj << " " << std::setprecision(12) << std::scientific
                         << u_x << " " << u_y
                         << " " << u << " " << pressure << " "
                         << (int) obstacles.at(ii, jj)
                         << std::endl;
                }
            }
            file.close();
            return true;
        }
        return false;
    }


    auto writeResultsAoS(const std::string &filename,
                         const Params &params,
                         const Obstacles &obstacles,
                         const CellsSoA &cells) -> bool {
        const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
        float local_density;         /* per grid cell sum of densities */
        float pressure;              /* fluid pressure in grid cell */
        float u_x;                   /* x-component of velocity in grid cell */
        float u_y;                   /* y-component of velocity in grid cell */
        float u;                     /* norm--root of summed squares--of u_x and u_y */

        std::ofstream file;
        file.open(filename, std::ios::out);
        if (file.is_open()) {

            const auto nx = params.nx;
            const auto ny = params.ny;
            for (auto jj = 0u; jj < ny; jj++) {
                for (auto ii = 0u; ii < nx; ii++) {
                    /* an occupied cell */
                    if (obstacles.at(ii, jj)) {
                        u_x = u_y = u = 0.f;
                        pressure = params.density * c_sq;
                    }
                        /* no obstacle */
                    else {
                        const auto idx = jj * nx + ii;

                        local_density = cells.middleSpeeds[idx] + cells.northSpeeds[idx] + cells.southSpeeds[idx] +
                                        cells.eastSpeeds[idx] + cells.westSpeeds[idx] + cells.northWestSpeeds[idx] +
                                        cells.northEastSpeeds[idx] + cells.southWestSpeeds[idx] +
                                        cells.southEastSpeeds[idx];

                        /* x-component of velocity */
                        float u_x = (cells.eastSpeeds[idx] + cells.northEastSpeeds[idx] + cells.southEastSpeeds[idx] -
                                     (cells.westSpeeds[idx] + cells.northWestSpeeds[idx] +
                                      cells.southWestSpeeds[idx])) /
                                    local_density;
                        /* compute y velocity component */
                        float u_y = (cells.northSpeeds[idx] + cells.northEastSpeeds[idx] + cells.northWestSpeeds[idx] -
                                     (cells.southEastSpeeds[idx] + cells.southWestSpeeds[idx] +
                                      cells.southSpeeds[idx])) /
                                    local_density;

                        /* compute norm of velocity */
                        u = sqrtf((u_x * u_x) + (u_y * u_y));
                        /* compute pressure */
                        pressure = local_density * c_sq;
                    }

                    /* write to file */
                    file << ii << " " << jj << " " << std::setprecision(12) << std::scientific
                         << u_x << " " << u_y
                         << " " << u << " " << pressure << " "
                         << (int) obstacles.at(ii, jj)
                         << std::endl;
                }
            }
            file.close();
            return true;
        }
        return false;
    }


};


#endif //LBM_GRAPHCORE_LATTICEBOLTZMANN_H
