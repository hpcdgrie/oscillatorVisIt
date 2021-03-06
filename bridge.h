#ifndef OSCILLATORS_BRIDGE_H
#define OSCILLATORS_BRIDGE_H

#include <mpi.h>
#include <string>

#include "Particles.h"

class Bridge
{

public:
  Bridge(const MPI_Comm &communicator);
  int initialize(size_t nblocks, size_t n_local_blocks, float *origin,
                 float *spacing, int domain_shape_x, int domain_shape_y, int domain_shape_z,
                 int *gid, int *from_x, int *from_y, int *from_z, int *to_x, int *to_y,
                 int *to_z, int *shape, int ghostLevels, const std::string &config_file);

  void set_data(int gid, float *data);
  void set_particles(int gid, const std::vector<Particle> &particles);

  bool execute(long step, float time);

  void finalize();

  int rank = 0;
  int size = 0;
  struct InternalsType;
  InternalsType *Internals;
private:
  const MPI_Comm &comm;

};

#endif
