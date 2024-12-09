#include <fstream>
#include <iomanip>
#include "Array.hpp"

void float_to_bin(const PointArray& pts, unsigned char depth, BinaryArray& out)
{
    const unsigned int N_ = (1 << depth);

    #pragma omp parallel for
    for (int i = 0; i < pts.shape[0]; i++)
    {
        for (unsigned int j = 0; j < pts.shape[1]; j++)
        {
            for (unsigned char d = 0; d < depth; d++)
            {
                const unsigned int pointValue = pts[{(uint32_t)i, j}] * N_;
                const unsigned int po2 = pointValue & (1 << (depth - d - 1));
                out.data[out.indexOf({(uint32_t)i, j, d})] = (po2 != 0);
            }
        }
    }
}

void bin_to_float(const FuzzyBinaryArray& array, PointArray& outArray)
{
    #pragma omp parallel for
    for (int i = 0; i < array.shape[0]; i++)
    {
        for (unsigned int j = 0; j < array.shape[1]; j++)
        {
            outArray[{(uint32_t)i, j}] = 0.;
            for (unsigned int d = 0; d < array.shape[2]; d++)
            {
                const double po2 = 1.0 / (1 << (d + 1));
                outArray[{(uint32_t)i, j}] += po2 * array[{(uint32_t)i, j, d}];
            }
        }
    }
}

