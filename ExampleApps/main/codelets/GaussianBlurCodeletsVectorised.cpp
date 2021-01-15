#include <poplar/Vertex.hpp>
#include <cstddef>
#include <print.h>
#include <math.h>
#include <ipudef.h>

using namespace poplar;


//------------------------------------- FLOAT2 Implementation ---------------------------------------------------------
/**
 * Perform the Gaussian blur on 2 channels (or pixels of the same channel) simultaneously
 */
float2 stencil(const float2 nw, const float2 n, const float2 ne, const float2 w, const float2 m,
               const float2 e, const float2 sw,
               const float2 s, const float2 se) {
    return 1.f / 16 * (nw + ne + sw + se) + 4.f / 16 * m + 2.f / 16 * (e + w + s + n);
}

/** Recast Input/Output as a float2 * to generate 64-bit loads and stores */
#define AS_F2(X)    reinterpret_cast<float2 *>(&X[0])
/** The index in the float2 array of the current (x,y) index offset by (R,C) items */
#define F2_OUTIDX(R, C) 2 * (width * (y + R) + (x + C)) + c
#define F2_INIDX(ROW, COL)    2 * ((width +2)* ((y + 1) + ROW) + ((x + 1) + COL)) + c

class GaussianBlurCodeletFloat2 : public Vertex {

public:
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> in;
    Output <Vector<float, VectorLayout::ONE_PTR, 8>> out;

    unsigned width;
    unsigned height;

    bool compute() {
        const auto f2in = AS_F2(in);
        auto f2out = AS_F2(out);
        for (auto y = 0; y < height; y++) {
            for (auto x = 0; x < width; x++) {
#pragma unroll 2
                for (auto c = 0u; c < 2; c++) {
                    const auto _nw = f2in[F2_INIDX(-1, -1)];
                    const auto _w = f2in[F2_INIDX(0, -1)];
                    const auto _sw = f2in[F2_INIDX(+1, -1)];
                    const auto _n = f2in[F2_INIDX(-1, 0)];
                    const auto _m = f2in[F2_INIDX(0, 0)];
                    const auto _s = f2in[F2_INIDX(+1, 0)];
                    const auto _ne = f2in[F2_INIDX(+1, +1)];
                    const auto _e = f2in[F2_INIDX(0, +1)];
                    const auto _se = f2in[F2_INIDX(-1, +1)];
                    f2out[F2_OUTIDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }
        }
        return true;
    }
};


/** Recast Input/Output as a half4 * to generate 64-bit loads and stores */
#define AS_H4(X)    reinterpret_cast<half4 *>(&X[0])
/** The index in the half4 array of the current (x,y) index offset by (R,C) items */
#define H4_OUTIDX(R, C) (width * (y + R) + (x + C))
#define H4_INIDX(ROW, COL)  (width +2)* ((y + 1) + ROW) + ((x + 1) + COL)

half4 stencil(const half4 nw, const half4 n, const half4 ne, const half4 w, const half4 m,
               const half4 e, const half4 sw,
               const half4 s, const half4 se) {
    return 1.f / 16 * (nw + ne + sw + se) + 4.f / 16 * m + 2.f / 16 * (e + w + s + n);
}

class GaussianBlurCodeletHalf4 : public Vertex {

public:
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> in;
    Output <Vector<half, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height;

    bool compute() {
        const auto h4in = AS_H4(in);
        auto h4out = AS_H4(out);

        for (auto y = 0; y < height ; y++) {
            for (auto x = 0; x < width; x++) {
                const auto _nw = h4in[H4_INIDX(-1, -1)];
                const auto _w = h4in[H4_INIDX(0, -1)];
                const auto _sw = h4in[H4_INIDX(+1, -1)];
                const auto _n = h4in[H4_INIDX(-1, 0)];
                const auto _m = h4in[H4_INIDX(0, 0)];
                const auto _s = h4in[H4_INIDX(+1, 0)];
                const auto _ne = h4in[H4_INIDX(+1, +1)];
                const auto _e = h4in[H4_INIDX(0, +1)];
                const auto _se = h4in[H4_INIDX(-1, +1)];
                h4out[H4_OUTIDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
            }
        }
        return true;
    }
};
