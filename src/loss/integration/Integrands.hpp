#pragma once

#include <istream>

// Important note : 
//  - The ground truth reprensent the ground truth of the perfect heaviside (ie: when SmoothFactor -> +oo)
struct SoftHeavisideIntegrands
{
    static double SmoothFactor;

    // Non part of the function
    static unsigned int ComputeNumParams(unsigned int dim);
    static void ReadData(std::istream& in, unsigned int dim, double* dest);
    static double Evaluate(
        unsigned int dim, 
        const double*  pts, 
        const double* data, 
        double* grad
    );
};

struct GaussiansIntegrands
{
    // Non part of the function
    static unsigned int ComputeNumParams(unsigned int dim);
    static void ReadData(std::istream& in, unsigned int dim, double* dest);
    static double Evaluate(
        unsigned int dim, 
        const double*  pts, 
        const double* data, 
        double* grad
    );
};

