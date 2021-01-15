#include <poplar/Vertex.hpp>
#include <cstddef>
#include <print.h>
#include <math.h>
#include <ipudef.h>

using namespace poplar;

/**
 * Gaussian Blur on 4 channels (RGBA) of the input image, unvectorised implementation
 */

auto constexpr NumChannels = 4;

// We can't template this because we aren't allow half params
#define sixteenth ((T) 1.f / 16)
#define quarter ((T) 4.f / 16)
#define eighth ((T) 2.f / 16)
#define stencil(nw, n, ne, w, m, e, sw, s, se) sixteenth * (nw + ne + sw + se) + quarter * m + eighth * (e + w + s + n);

template<typename T>
class GaussianBlurCodelet : public Vertex {

#define INIDX(ROW, COL)   NumChannels * ((width +2)* ((y + 1) + ROW) + ((x + 1) + COL)) + c
#define OUTIDX(ROW, COL)   NumChannels * (width * (y + ROW) + (x + COL)) + c

public:
    Input <Vector<T, VectorLayout::ONE_PTR>> in;
    Output <Vector<T, VectorLayout::ONE_PTR>> out;
    unsigned width;
    unsigned height;

    bool compute() {
        const T *inPtr = reinterpret_cast<T *>(&in[0]);
        T *outPtr = reinterpret_cast<T *>(&out[0]);
        for (auto y = 0u; y < height ; y++) {
            for (auto x = 0u; x < width ; x++) {
                for (auto c = 0; c < NumChannels; c++) {
                    outPtr[OUTIDX(0, 0)] = stencil(
                            inPtr[INIDX(-1, -1)], inPtr[INIDX(-1, 0)], inPtr[INIDX(-1, 1)],
                            inPtr[INIDX(0, -1)], inPtr[INIDX(0, 0)], inPtr[INIDX(0, 1)],
                            inPtr[INIDX(1, -1)], inPtr[INIDX(1, 0)], inPtr[INIDX(1, 1)]
                    );
                }
            }
        }
        return true;
    }
};

template
class GaussianBlurCodelet<float>;


template
class GaussianBlurCodelet<half>;
