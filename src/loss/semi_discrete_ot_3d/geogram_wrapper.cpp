#include "geogram_wrapper.hpp"

#ifdef WITH_GEOGRAM

#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/stopwatch.h>
#include <geogram/basic/file_system.h>
#include <geogram/basic/process.h>
#include <geogram/basic/progress.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_tetrahedralize.h>
#include <geogram/voronoi/CVT.h>

#include <exploragram/optimal_transport/optimal_transport_3d.h>
#include <exploragram/optimal_transport/sampling.h>

void Init(int argc, char** argv)
{
    ((void) argc); ((void) argv);

    static bool geogram_init = false;
    using namespace GEO;

    if (!geogram_init)
    {
        GEO::initialize();
        CmdLine::import_arg_group("standard");
        CmdLine::import_arg_group("algo");
        CmdLine::import_arg_group("opt");

        geogram_init = false;
    }
}

bool load_volume_mesh(const std::string& filename, GEO::Mesh& M)
{
    Init(0, nullptr);

    using namespace GEO;

    MeshIOFlags flags;
    flags.set_element(MESH_CELLS);
    flags.set_attribute(MESH_CELL_REGION);

    if(!mesh_load(filename, M, flags)) {
        return false;
    }

    return true;
}

void OptimalTransport(GEO::Mesh& m, unsigned int N, const double* pts, double* centroids)
{
    GEO::compute_Laguerre_centroids_3d(
        &m, N, pts, centroids
    );
}

Density3D::Density3D(const std::string& filename)
{
    mesh = new GEO::Mesh();
    load_volume_mesh(filename, *mesh);
}

Density3D::~Density3D()
{
    delete mesh;
}

Transport3D::Transport3D(const std::string& filename) :
    otmap(nullptr), density(filename)
{
    density.mesh->vertices.set_dimension(4); // Required by library

    otmap = new GEO::OptimalTransportMap3d(density.mesh);
    otmap->set_regularization(1e-3);
    otmap->set_Newton(true);
    otmap->set_epsilon(0.01);
    otmap->set_verbose(false);

}

double Transport3D::Compute(unsigned int N, const double* pts, double* centroids) const
{
    otmap->set_points(N, pts);
    otmap->set_Laguerre_centroids(centroids);
    otmap->optimize(100);

    return otmap->getValue();
}

Transport3D::~Transport3D()
{
    delete otmap;
}

#endif