#pragma once

#include "Array.hpp"

void float_to_bin(const PointArray& pts, unsigned char depth, BinaryArray& out);

void bin_to_float(const FuzzyBinaryArray& array, PointArray& outArray);
