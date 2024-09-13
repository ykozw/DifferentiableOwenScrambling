#pragma once

#include "utils/Array.hpp"

double gbn_toroidal_forward_backward (const PointArray& array, PointArray& grad, double sigma);
double gbn_euclidean_forward_backward(const PointArray& array, PointArray& grad, double sigma);
