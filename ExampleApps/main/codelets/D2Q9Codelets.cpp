#include <poplar/Vertex.hpp>
#include <cstddef>
#include <print.h>
#include <math.h>
#include <ipudef.h>

/*
 * Slightly unintuitively we are using (0,0) as the bottom left corner
 */

using namespace poplar;

/**
 *  The output array of summed velocities is spread throughout the distributed memories. We give each
 *  tile a vertex that knows what bits of the array are mapped to it. The sumOfVelocities is broadcast to
 *  each tile, and only the tile owning the memory writes it.
 */
class AppendReducedSum : public Vertex
{ // Reduce the per-tile partial sums and append to the list of sums

public:
    Input<float> sumOfVelocities;
    Input<unsigned> indexToWrite;
    unsigned myStartIndex;       // The index where my array starts
    unsigned myEndIndex;         // My last index (inclusive)
    InOut<Vector<float>> finals; // The piece of the array I have

    bool compute()
    {

        const auto idx = *indexToWrite;
        if ((idx >= myStartIndex) && (idx <= myEndIndex))
        {
            finals[idx - myStartIndex] = sumOfVelocities;
        }
        return true;
    }
};

class IncrementIndex : public Vertex
{
public:
    InOut<unsigned> index;

    auto compute() -> bool
    {
        (*index)++;
        return true;
    }
};

struct Cell
{
    float speeds[9];
};

struct Params
{
    int ny;
    int nx;
    int maxIters;
    float omega;
    float one_minus_omega;
    float density;
    float accel;
    bool isAccelerate;
    int rowToAccelerate;
    int total_free_cells;
};

void accelerateRow(const Params &params, const bool *obstacles, Cell *cells)
{
    const float w1 = params.density * params.accel / 9.f;
    const float w2 = params.density * params.accel / 36.f;
    for (int ii = 0; ii < params.nx; ii++)
    {
        /* if the cell is not occupied and
        ** we don't send a negative density */
        if (!obstacles[ii] && (cells[ii].speeds[3] - w1) > 0.f &&
            (cells[ii].speeds[6] - w2) > 0.f &&
            (cells[ii].speeds[7] - w2) > 0.f)
        {
            /* increase 'east-side' densities */
            cells[ii].speeds[1] += w1;
            cells[ii].speeds[5] += w2;
            cells[ii].speeds[8] += w2;
            /* decrease 'west-side' densities */
            cells[ii].speeds[3] -= w1;
            cells[ii].speeds[6] -= w2;
            cells[ii].speeds[7] -= w2;
        }
    }
}
auto lbmKernel(const Params &params, const Cell *cells_old, Cell *cells_new, const bool *obstacles) -> float
{
    /* compute weighting factors */
    const float w1 = params.density * params.accel / 9.f;
    const float w2 = params.density * params.accel / 36.f;
    float tot_u = 0.00f;

    // Old is ny rows x nx cols
#define NEW_OFFSET(r, c) (ii + c) + (jj + r) * params.nx
#define OLD_OFFSET(r, c) (ii + 1 + c) + (jj + 1 + r) * (params.nx + 2)
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            const int y_n = (jj + 1) % params.ny;
            const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
            const float accel = jj == params.rowToAccelerate && params.isAccelerate ? 1.f : 0.f;
            const int x_e = (ii + 1) % params.nx;
            const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
            const int is_obstacle = obstacles[NEW_OFFSET(0, 0)];

            const float speeds_0 = cells_old[OLD_OFFSET(+0, +0)].speeds[0]; /* central cell, no movement */
            const float speeds_1 = cells_old[OLD_OFFSET(+0, -1)].speeds[1]; /* east */
            const float speeds_2 = cells_old[OLD_OFFSET(-1, +0)].speeds[2]; /* north */
            const float speeds_3 = cells_old[OLD_OFFSET(+0, +1)].speeds[3]; /* west */
            const float speeds_4 = cells_old[OLD_OFFSET(+1, +0)].speeds[4]; /* south */
            const float speeds_5 = cells_old[OLD_OFFSET(-1, -1)].speeds[5]; /* north-east */
            const float speeds_6 = cells_old[OLD_OFFSET(-1, +1)].speeds[6]; /* north-west */
            const float speeds_7 = cells_old[OLD_OFFSET(+1, +1)].speeds[7]; /* south-west */
            const float speeds_8 = cells_old[OLD_OFFSET(+1, -1)].speeds[8]; /* south-east */

            if (is_obstacle)
            {
                cells_new[NEW_OFFSET(0, 0)].speeds[0] = speeds_0;
                cells_new[NEW_OFFSET(0, 0)].speeds[1] = speeds_3;
                cells_new[NEW_OFFSET(0, 0)].speeds[2] = speeds_4;
                cells_new[NEW_OFFSET(0, 0)].speeds[3] = speeds_1;
                cells_new[NEW_OFFSET(0, 0)].speeds[4] = speeds_2;
                cells_new[NEW_OFFSET(0, 0)].speeds[5] = speeds_7;
                cells_new[NEW_OFFSET(0, 0)].speeds[6] = speeds_8;
                cells_new[NEW_OFFSET(0, 0)].speeds[7] = speeds_5;
                cells_new[NEW_OFFSET(0, 0)].speeds[8] = speeds_6;
            }
            else
            {

                /* compute local density total */
                const float local_density =
                    speeds_0 + speeds_1 + speeds_2 +
                    speeds_3 + speeds_4 + speeds_5 +
                    speeds_6 + speeds_7 + speeds_8;

                /* compute x velocity component */
                const float u_x = (speeds_1 + speeds_5 + speeds_8 -
                                   (speeds_3 + speeds_6 + speeds_7)) /
                                  local_density;
                /* compute y velocity component */
                const float u_y = (speeds_2 + speeds_5 + speeds_6 -
                                   (speeds_4 + speeds_7 + speeds_8)) /
                                  local_density;

                /* velocity squared */
                const float u_sq = u_x * u_x + u_y * u_y;

                const float c_sq = 1.00f - u_sq * 1.50f;
                const float ld0 = 4.00f / 9.00f * local_density * params.omega;
                const float ld1 = local_density / 9.00f * params.omega;
                const float ld2 = local_density / 36.00f * params.omega;
                const float u_s = u_x + u_y;
                const float u_d = -u_x + u_y;

                const auto relax2 = [&](const float2 speed, const float weight, const float2 velocityComponent) -> float2 {
                    return speed * params.one_minus_omega +
                           weight * ((4.50f * velocityComponent) * (2.00f / 3.00f + velocityComponent) + c_sq);
                };

                auto [speeds_out_3, speeds_out_1] = relax2(float2{speeds_3, speeds_1}, ld1, float2{-u_x, u_x});
                auto [speeds_out_4, speeds_out_2] = relax2(float2{speeds_4, speeds_2}, ld1, float2{-u_y, u_y});
                auto [speeds_out_6, speeds_out_5] = relax2(float2{speeds_6, speeds_5}, ld2, float2{-u_x + u_y, u_x + u_y});
                auto [speeds_out_7, speeds_out_8] = relax2(float2{speeds_7, speeds_8}, ld2, float2{-u_x - u_y, u_x - u_y});

                const float speeds_out_0 = speeds_0 * params.one_minus_omega + ld0 * c_sq;
                
                cells_new[NEW_OFFSET(0, 0)].speeds[0] = speeds_out_0;
                cells_new[NEW_OFFSET(0, 0)].speeds[1] = speeds_out_1 + accel * w1;
                cells_new[NEW_OFFSET(0, 0)].speeds[2] = speeds_out_2;
                cells_new[NEW_OFFSET(0, 0)].speeds[3] = speeds_out_3 - accel * w1;
                cells_new[NEW_OFFSET(0, 0)].speeds[4] = speeds_out_4;
                cells_new[NEW_OFFSET(0, 0)].speeds[5] = speeds_out_5 + accel * w2;
                cells_new[NEW_OFFSET(0, 0)].speeds[6] = speeds_out_6 - accel * w2;
                cells_new[NEW_OFFSET(0, 0)].speeds[7] = speeds_out_7 - accel * w2;
                cells_new[NEW_OFFSET(0, 0)].speeds[8] = speeds_out_8 + accel * w2;
                tot_u += sqrtf(u_sq);
            }
        }
    }
    return tot_u / (float)params.total_free_cells;
}

class FirstAccelerateVertex : public Vertex
{

public:
    InOut<Vector<float, VectorLayout::ONE_PTR>> cellsVec;
    Input<Vector<bool, VectorLayout::ONE_PTR>> obstaclesVec;
    int nx;
    float density;
    float accel;

    bool compute()
    {
        auto cells = reinterpret_cast<Cell *>(&cellsVec[0]);
        auto obstacles = reinterpret_cast<bool *>(&obstaclesVec[0]);

        auto params = Params{
            .ny = 0,
            .nx = nx,
            .maxIters = 0,
            .omega = 0,
            .one_minus_omega = 0,
            .density = density,
            .accel = accel,
            .isAccelerate = true,
            .rowToAccelerate = 0,
            .total_free_cells = 0};

        accelerateRow(params, obstacles, cells);

        return true;
    }
};

class LbmTimeStepVertex : public Vertex
{

public:
    Input<Vector<float, VectorLayout::ONE_PTR>> cells_oldVec;
    Output<Vector<float, VectorLayout::ONE_PTR>> cells_newVec;
    Input<Vector<bool, VectorLayout::ONE_PTR>> obstaclesVec;
    Output<float> av_vel;
    int ny;
    int nx;
    int maxIters;
    float omega;
    float one_minus_omega;
    float density;
    float accel;
    float iter;
    int total_free_cells;
    bool isAccelerate;
    int rowToAccelerate;

    bool compute()
    {
        auto cells_old = reinterpret_cast<Cell *>(&cells_oldVec[0]);
        auto cells_new = reinterpret_cast<Cell *>(&cells_newVec[0]);
        auto obstacles = reinterpret_cast<bool *>(&obstaclesVec[0]);

        auto params = Params{
            .ny = ny,
            .nx = nx,
            .maxIters = maxIters,
            .omega = omega,
            .one_minus_omega = one_minus_omega,
            .density = density,
            .accel = accel,
            .isAccelerate = isAccelerate,
            .rowToAccelerate = rowToAccelerate,
            .total_free_cells = total_free_cells};

        *av_vel = lbmKernel(params, cells_old, cells_new, obstacles);

        return true;
    }
};
