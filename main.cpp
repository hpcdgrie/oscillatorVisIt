#include <vector>
#include <chrono>
#include <ctime>

#include <diy/master.hpp>
#include <diy/decomposition.hpp>
#include <diy/io/bov.hpp>
#include <diy/grid.hpp>
#include <diy/vertices.hpp>
#include <diy/point.hpp>

#include "Oscillator.h"
#include "bridge.h"
#include "Particles.h"
#include "Block.h"
#include <mpi.h>
using Grid = diy::Grid<float, 3>;
using Vertex = Grid::Vertex;
using SpPoint = diy::Point<float, 3>;
using Bounds = diy::Point<float, 6>;

using Link = diy::RegularGridLink;
using Master = diy::Master;
using Proxy = Master::ProxyWithLink;

using RandomSeedType = std::default_random_engine::result_type;

using Time = std::chrono::high_resolution_clock;
using ms = std::chrono::milliseconds;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: oscillator numGhost inputFile" << std::endl;
        return 1;
    }

    int required = MPI_THREAD_MULTIPLE;
    int provided = 0;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < required)
    {
        std::cerr << "This MPI does not support thread serialized" << std::endl;
        return 1;
    }
    auto start = Time::now();

    //diy::mpi::environment     env(argc, argv);
    diy::mpi::communicator world;
    {

        auto infn = argv[2];
        Vertex shape = {64, 64, 64};
        int nblocks = world.size();
        float t_end = 10;
        float dt = .01;
        float velocity_scale = 50.0f;
#ifndef ENABLE_SENSEI
        size_t window = 10;
        size_t k_max = 3;
#endif
        int threads = 1;
        int ghostCells = atoi(argv[1]);
        int numberOfParticles = 0;
        int seed = 0x240dc6a9;
        std::string config_file;
        std::string out_prefix = "";
        Bounds bounds{0., -1., 0., -1., 0., -1.};
        bool verbose = true;
        if (numberOfParticles < 0)
            numberOfParticles = nblocks * 64;

        int particlesPerBlock = numberOfParticles / nblocks;

        if (verbose && (world.rank() == 0))
        {
            std::cerr << world.rank() << " numberOfParticles = " << numberOfParticles << std::endl
                      << world.rank() << " particlesPerBlock = " << particlesPerBlock << std::endl;
        }

        if (seed == -1)
        {
            if (world.rank() == 0)
            {
                seed = static_cast<int>(std::time(nullptr));
                if (verbose)
                    std::cerr << world.rank() << " seed = " << seed << std::endl;
            }

            diy::mpi::broadcast(world, seed, 0);
        }

        std::default_random_engine rng(static_cast<RandomSeedType>(seed));
        for (int i = 0; i < world.rank(); ++i)
            rng(); // different seed for each rank

        std::vector<Oscillator> oscillators;
        if (world.rank() == 0)
        {
            oscillators = read_oscillators(infn);
            diy::MemoryBuffer bb;
            diy::save(bb, oscillators);
            diy::mpi::broadcast(world, bb.buffer, 0);
        }
        else
        {
            diy::MemoryBuffer bb;
            diy::mpi::broadcast(world, bb.buffer, 0);
            diy::load(bb, oscillators);
        }

        if (verbose && (world.rank() == 0))
            for (auto &o : oscillators)
                std::cerr << world.rank() << " center = " << o.center
                          << " radius = " << o.radius << " omega0 = " << o.omega0
                          << " zeta = " << o.zeta << std::endl;

        diy::Master master(world, threads, -1,
                           &Block::create,
                           &Block::destroy);

        diy::ContiguousAssigner assigner(world.size(), nblocks);

        diy::DiscreteBounds domain{3};
        domain.min[0] = domain.min[1] = domain.min[2] = 0;
        for (unsigned i = 0; i < 3; ++i)
            domain.max[i] = shape[i] - 1;

        SpPoint origin{0., 0., 0.};
        SpPoint spacing{1., 1., 1.};
        if (bounds[1] >= bounds[0])
        {
            // valid bounds specififed on the command line, calculate the
            // global origin and spacing.
            for (int i = 0; i < 3; ++i)
                origin[i] = bounds[2 * i];

            for (int i = 0; i < 3; ++i)
                spacing[i] = (bounds[2 * i + 1] - bounds[2 * i]) / shape[i];
        }

        if (verbose && (world.rank() == 0))
        {
            std::cerr << world.rank() << " domain = " << domain.min
                      << ", " << domain.max << std::endl
                      << world.rank() << "bounds = " << bounds << std::endl
                      << world.rank() << "origin = " << origin << std::endl
                      << world.rank() << "spacing = " << spacing << std::endl;
        }

        // record various parameters to initialize analysis
        std::vector<int> gids,
            from_x, from_y, from_z,
            to_x, to_y, to_z;

        diy::RegularDecomposer<diy::DiscreteBounds>::BoolVector share_face;
        diy::RegularDecomposer<diy::DiscreteBounds>::BoolVector wrap(3, true);
        diy::RegularDecomposer<diy::DiscreteBounds>::CoordinateVector ghosts = {ghostCells, ghostCells, ghostCells};

        // decompose the domain
        diy::decompose(
            3, world.rank(), domain, assigner,
            [&](int gid, const diy::DiscreteBounds &, const diy::DiscreteBounds &bounds,
                const diy::DiscreteBounds &domain, const Link &link)
            {
                Block *b = new Block(gid, bounds, domain, origin,
                                     spacing, ghostCells, oscillators, velocity_scale);

                // generate particles
                int start = particlesPerBlock * gid;
                int count = particlesPerBlock;
                if ((start + count) > numberOfParticles)
                    count = std::max(0, numberOfParticles - start);

                b->particles = GenerateRandomParticles<float>(rng,
                                                              b->domain, b->bounds, b->origin, b->spacing, b->nghost,
                                                              start, count);

                master.add(gid, b, new Link(link));

                gids.push_back(gid);

                from_x.push_back(bounds.min[0]);
                from_y.push_back(bounds.min[1]);
                from_z.push_back(bounds.min[2]);

                to_x.push_back(bounds.max[0]);
                to_y.push_back(bounds.max[1]);
                to_z.push_back(bounds.max[2]);

                if (verbose)
                    std::cerr << world.rank() << " Block " << *b << std::endl;
            },
            share_face, wrap, ghosts);
        Bridge bridge(MPI_COMM_WORLD);
        bridge.initialize(nblocks, gids.size(), origin.data(), spacing.data(),
                          domain.max[0] + 1, domain.max[1] + 1, domain.max[2] + 1,
                          &gids[0],
                          &from_x[0], &from_y[0], &from_z[0],
                          &to_x[0], &to_y[0], &to_z[0],
                          &shape[0], ghostCells,
                          config_file);

        int t_count = 0;
        float t = 0.;
        while (t < t_end)
        {
            if (verbose && (world.rank() == 0))
                std::cerr << "started step = " << t_count << " t = " << t << std::endl;

            master.foreach ([=](Block *b, const Proxy &)
                            { b->update_fields(t); });

            master.foreach ([=](Block *b, const Proxy &p)
                            { b->move_particles(dt, p); });

            master.exchange();

            master.foreach ([=](Block *b, const Proxy &p)
                            { b->handle_incoming_particles(p); });

            // do the analysis using sensei
            // update data adaptor with new data
            master.foreach ([&bridge](Block *b, const Proxy &)
                            {
                                bridge.set_data(b->gid, b->grid.data());
                                std::cerr << "setting data for block " << b->gid << " with size " << b->grid.size() << std::endl;
                                bridge.set_particles(b->gid, b->particles);
                            });
            // push data to sensei
            if (!bridge.execute(t_count, t))
                break;

            if (!out_prefix.empty())
            {
                auto out_start = Time::now();

                // Save the corr buffer for debugging
                std::ostringstream outfn;
                outfn << out_prefix << "-" << t << ".bin";

                diy::mpi::io::file out(world, outfn.str(), diy::mpi::io::file::wronly | diy::mpi::io::file::create);
                diy::io::BOV writer(out, shape);
                master.foreach ([&writer](Block *b, const diy::Master::ProxyWithLink &cp)
                                {
                                    auto link = static_cast<Link *>(cp.link());
                                    writer.write(link->bounds(), b->grid.data(), true);
                                });
                if (verbose && (world.rank() == 0))
                {
                    auto out_duration = std::chrono::duration_cast<ms>(Time::now() - out_start);
                    std::cerr << "Output time for " << outfn.str() << ":" << out_duration.count() / 1000
                              << "." << out_duration.count() % 1000 << " s" << std::endl;
                }
            }

            if (sync)
                world.barrier();

            t += dt;
            ++t_count;
        }
        bridge.finalize();
    }
    world.barrier();
    if (world.rank() == 0)
    {
        auto duration = std::chrono::duration_cast<ms>(Time::now() - start);
        std::cerr << "Total run time: " << duration.count() / 1000
                  << "." << duration.count() % 1000 << " s" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
