#pragma once

#include <cmath>
#include "utils/BinaryUtils.hpp"

uint64_t CountScramblingCount(unsigned int Depth, unsigned int D);
void SetOwenScrambling(uint64_t idx, unsigned int Depth, std::vector<Params>& params);

void FillRandomLeftoverBits(PointArray& dest, unsigned int seed, unsigned int DepthMin, unsigned int DepthMax);
void ApplyLeftoverBits(PointArray& dest, const PointArray& bits);

class SmoothOwenScrambling
{
public:
    SmoothOwenScrambling(unsigned int D, unsigned char Depth, double alph);

    std::pair<double, double> DFF(bool bit, double param);

    void Randomize(unsigned long long seed, unsigned long long channel = 1);

    void forward(const BinaryArray& pts, PointArray& out);

    // Apply backpropagation given gradient of the loss to the points
    void backward(const PointArray& gradients, double lr);

    void backwardStore(const PointArray& gradients, double lr, PointArray& grads);

    // Evaluate array
    void evaluate(const BinaryArray& pts, PointArray& out, unsigned int override_depth = 0);

    // Allocate gradients (and compute indices) for a special binary array
    void AllocateGradients(const BinaryArray& pts);
    
    uint32_t NumParamsPerDims() const;

    void Export(const std::string& filename) const;

    void ExportGrad(const std::string& filename, const BinaryArray& pts, unsigned int dim);
public: // Easy access to thetas
    double alpha;
    unsigned int Depth;
  
    // Parameters
    std::vector<Params> thetas;
    
    // {D, 2 ** Depth, ?}: maps parameters to the point it influcences
    std::vector<std::vector<std::vector<unsigned int>>> gradientsIndices;
    // {D, 2 ** Depth, N}: values for each parameter for each point
    std::vector<std::vector<std::vector<double>>>       gradientsValues;
    
};
