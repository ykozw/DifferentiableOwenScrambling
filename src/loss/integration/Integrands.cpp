#include "Integrands.hpp"
#include <cmath>

double SoftHeavisideIntegrands::SmoothFactor = 2.0;

unsigned int SoftHeavisideIntegrands::ComputeNumParams(unsigned int dim)
{
    // Exclude gt !
    return dim + 1;
}

void SoftHeavisideIntegrands::ReadData(std::istream& in, unsigned int dim, double* dest)
{
    const unsigned int to_read = ComputeNumParams(dim);
    for (unsigned int i = 0; i < to_read; i++) in >> dest[i];
}

double SoftHeavisideIntegrands::Evaluate(
    unsigned int   dim,
    const double*  pts, 
    const double* data, 
    double* grad
)
{
    double pointValue = data[0];
    for (unsigned int j = 0; j < dim; j++)
        pointValue += pts[j] * data[j + 1];                
    
    // Sigmoid'(alpha <a, x>) =  alpha * a * Sigmoid * (1 - Sigmoid)
    pointValue = 1.0 / (1 + std::exp(-SmoothFactor * pointValue));
    
    // Avoid high derivatives around sigmoid's tipping point
    //  const double derivative = std::min(alpha * pointValue * (1 - pointValue), threshold);
    // Leave derivatives as-is
    const double derivative = 2 * SmoothFactor * pointValue * (1 - pointValue);
    for (unsigned int j = 0; j < dim; j++)
        grad[j] = data[j + 1] * derivative;

    return pointValue;
}


unsigned int GaussiansIntegrands::ComputeNumParams(unsigned int dim)
{
    // Exclude gt !
    return dim + dim * dim;
}

void GaussiansIntegrands::ReadData(std::istream& in, unsigned int dim, double* dest)
{
    const unsigned int to_read = ComputeNumParams(dim);
    for (unsigned int i = 0; i < to_read; i++) in >> dest[i];
}

double GaussiansIntegrands::Evaluate(
    unsigned int   dim,
    const double*  pts, 
    const double* data, 
    double* grad
)
{
    const double* mu       = data + 0;
    const double* invSigma = data + dim;

    double rslt = 0.0;
    for (unsigned int i = 0; i < dim; i++)
    {
        double tmp = 0.0;
        for (unsigned int j = 0; j < dim; j++)
            tmp += invSigma[j + i * dim] * (pts[j] - mu[j]);
        grad[i] = tmp;
        rslt += tmp * (pts[i] - mu[i]);
    }

    rslt = std::exp(-0.5 * rslt);

    // -0.5 (from exp) * 2 (from derivativative of <Ax, x>) = -1
    const double drslt = -rslt;
    for (unsigned int i = 0; i < dim; i++)
        grad[i] *= drslt;

    return rslt;
}
