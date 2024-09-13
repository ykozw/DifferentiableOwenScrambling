#pragma once

#include "utils/Array.hpp"
#include <vector>

class PCF
{
public:
    PCF(
        unsigned int N, unsigned int D,
        double sigma, 
        double ra, double rb, unsigned int nbins, 
        const std::string& targetFilename
    );

    double operator()(const PointArray& pts, PointArray& grad, std::vector<double>& out) const;
    const unsigned int N;
    const unsigned int D;
public:
    mutable PointArray tmpGrad;

    double sigma;
    double ra, rb;
    unsigned int nbins;

    std::vector<double> target;

};