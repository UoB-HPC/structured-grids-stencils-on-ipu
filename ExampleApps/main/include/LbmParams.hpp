//
// Created by Thorben Louw on 25/06/2020.
//

#ifndef LBM_GRAPHCORE_LBMPARAMS_H
#define LBM_GRAPHCORE_LBMPARAMS_H

#include <iostream>
#include <fstream>
#include <optional>
#include <vector>
#include <memory>

namespace lbm {

    class Params {
    public:
        const size_t nx;
        const size_t ny;
        const size_t maxIters;
        const size_t reynolds_dim;
        const float density;
        const float accel;
        const float omega;

        Params() = delete;

        static auto fromFile(const std::string &filename) -> std::optional<Params> {
            std::ifstream file;
            file.open(filename, std::ios::in);
            auto readUnsigned = [&file]() -> size_t {
                std::string line;
                std::getline(file, line);
                return stoul(line);
            };
            auto readFloat = [&file]() -> float {
                std::string line;
                std::getline(file, line);
                return stof(line);
            };
            if (file.is_open()) {

                //"nx", "ny", "maxIters", "reynolds_dim", "density", "accel", "omega"
                auto params = Params{
                        readUnsigned(),
                        readUnsigned(),
                        readUnsigned(),
                        readUnsigned(),
                        readFloat(),
                        readFloat(),
                        readFloat()
                };
                file.close();
                return {params};
            }
            std::cerr << "Could not read parameters from " << filename << std::endl;
            return std::nullopt;
        }

    private:
        Params(size_t nx, size_t ny, size_t maxIters, size_t reynolds_dim, float density, float accel, float omega) :
                nx(nx), ny(ny), maxIters(maxIters), reynolds_dim(reynolds_dim), density(density), accel(accel),
                omega(omega) {}

    };

    class Obstacles {
    public:
        const size_t nx;
        const size_t ny;
    private:

        std::unique_ptr<bool[]> data;


    public:

        Obstacles() = delete;

        auto getData() -> bool * {
            return data.get();
        };

        auto data_ptr() -> const std::unique_ptr<bool[]>& {
            return data;
        }

        [[nodiscard]] auto at(const size_t x, const size_t y) const -> bool {
            return data[y * nx + x];
        }

        static auto fromFile(size_t nx, size_t ny, const std::string &filename) -> std::optional<Obstacles> {
            auto data = std::unique_ptr<bool[]>{new bool[nx * ny]()};
            std::ifstream file;
            file.open(filename, std::ios::in);
            auto readLine = [&file]() -> std::optional<std::tuple<size_t, size_t>> {
                std::string line;
                if (std::getline(file, line)) {
                    int nx, ny, obstacle;
                    sscanf(line.c_str(), "%d %d %d", &nx, &ny, &obstacle);
                    if (obstacle == 1) {
                        return {std::make_tuple(nx, ny)};
                    } else {
                        std::cerr << "Malformed line: obstacle must be 1" << std::endl;
                        return std::nullopt;
                    }
                }
                return std::nullopt;
            };

            if (file.is_open()) {
                auto xy = readLine();
                while (xy.has_value()) {
                    const auto &[x, y] = xy.value();
                    data[y * nx + x] = true;
                    xy = readLine();
                }
                file.close();
                return {Obstacles(nx, ny, std::move(data))};
            }
            std::cerr << "Could not read parameters from " << filename << std::endl;
            return std::nullopt;
        }

    private:
        explicit Obstacles(const size_t nx, const size_t ny, std::unique_ptr<bool[]> data) :
                nx(nx), ny(ny), data(std::move(data)) {};
    };


};

#endif //LBM_GRAPHCORE_LBMPARAMS_H
