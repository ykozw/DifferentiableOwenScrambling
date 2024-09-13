#pragma once

#ifdef WITH_GEOGRAM

#include <geogram/mesh/mesh.h>
#include <exploragram/optimal_transport/optimal_transport_3d.h>

void Init(int argc, char** argv);
bool load_volume_mesh(const std::string& filename, GEO::Mesh& M);

struct Density3D
{
    Density3D(const std::string& filename);
    ~Density3D();
    
    GEO::Mesh* mesh;
};

struct Transport3D
{
public:
    Transport3D(const std::string& loadDensity);

    double Compute(unsigned int N, const double* pts, double* centroids) const;

    ~Transport3D();
private:
    double val;

    // Those are mutated by the lib for some reasons...
    mutable GEO::OptimalTransportMap3d* otmap;
    mutable Density3D density;
};

#else
    #include <string>

    namespace GEO { struct Mesh{}; struct OptimalTransportMap3d {}; }

    inline void Init(int argc, char** argv) { }
    inline bool load_volume_mesh(const std::string& filename, GEO::Mesh& M) { };

    struct Density3D
    {
        Density3D(const std::string& filename) {}
        ~Density3D() { };
        
        GEO::Mesh* mesh;
    };

    struct Transport3D
    {
    public:
        Transport3D(const std::string& loadDensity) : density(loadDensity) {}

        double Compute(unsigned int N, const double* pts, double* centroids) const { return 1; }

        ~Transport3D() {}
    private:
        // Those are mutated by the lib for some reasons...
        mutable GEO::OptimalTransportMap3d* otmap;
        mutable Density3D density;
    };
#endif