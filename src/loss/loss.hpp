#pragma once

#include <cmath>
#include "utils/BinaryUtils.hpp"
#include "screenspace/screenspace.hpp"
#include "screenspace/screenspace_reg.hpp"

struct Density2D
{
    static Density2D LoadBMP(const char* filepath);

    std::vector<double> data;
    unsigned int width;
    unsigned int height;
};

double  W2Loss2D(const PointArray& array, PointArray& grad);
double  W2Loss1pt2D(const PointArray& array, PointArray& grad);

double W2Loss3D(const PointArray& array, PointArray& grad);

// So: transport.h has no include guard and causes problem if included multiple times anyway...
void    SetNonUniformDensity2D(const Density2D& density);
double  NonUniformW2Loss2D(const PointArray& array, PointArray& grad, const Density2D& density);

void SetGBNSigma(double sigma);
double GBNLoss(const PointArray& array, PointArray& grad);