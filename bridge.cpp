#include "bridge.h"

#include <vector>
#include <experimental/filesystem>
#include <VisItDataInterface_V2.h>
#include <VisItControlInterface_V2.h>
#include <diy/master.hpp>

static Bridge *bridge;
struct Bridge::InternalsType
{
  InternalsType() : NumBlocks(0), Origin{},
                    Spacing{1, 1, 1}, Shape{}, NumGhostCells(0) {}

  long NumBlocks;                                   // total number of blocks on all ranks
  diy::DiscreteBounds DomainExtent{3};              // global index space
  std::map<long, diy::DiscreteBounds> BlockExtents; // local block extents, indexed by global block id
  std::map<long, float *> BlockData;                // local data array, indexed by block id
  std::map<long, const std::vector<Particle> *> ParticleData;

  double Origin[3];  // lower left corner of simulation domain
  double Spacing[3]; // mesh spacing

  int Shape[3];
  int NumGhostCells; // number of ghost cells
                     //-------------------------------------------------------
                     //stored to be catched from backend
                     //-------------------------------------------------------
  double time = 0;
  int cycle = 0;
  std::array<std::vector<float>, 3> vertices; //x, yand z array of the grid
  std::vector<int> cl;                        //connectivity list. Per element: type + vertex indices
  std::vector<char> ghostIndicees;
};
static int visit_broadcast_int_callback(int *value, int sender,
                                        void *ptr)
{
  return MPI_Bcast(value, 1, MPI_INT, sender, MPI_COMM_WORLD);
}
static int visit_broadcast_string_callback(char *str, int len, int sender, void *ptr)
{
  return MPI_Bcast(str, len, MPI_CHAR, sender, MPI_COMM_WORLD);
}

static void VisItSlaveProcessCallback()
{
  //int command = 0;
  //MPI_Bcast(&command, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

static int terminate = false;
static bool runflag = false;
static bool doAStep = false;
static void VisitCommandCallback(const char *cmd, const char *args, void *dbData)
{
  if (!strcmp(cmd, "stop"))
  {
    runflag = false;
  }
  else if (!strcmp(cmd, "step"))
  {
    doAStep = true;
  }
  else if (!strcmp(cmd, "run"))
  {
    runflag = true;
  }
  else if (!strcmp(cmd, "exit"))
  {
    terminate = true;
    std::cerr << "rank " << bridge->rank << ": disconnecting" << std::endl;
    VisItDisconnect();
  }
}

static int VisItActivateTimestep(void *ptr)
{
  return true;
}

static visit_handle VisItGetMetaData(void *ptr)
{
  visit_handle simHandle;

  VisIt_SimulationMetaData_alloc(&simHandle);
  VisIt_SimulationMetaData_setCycleTime(simHandle, bridge->Internals->cycle, bridge->Internals->time);

  std::array<const char *, 4> commands{"run", "stop", "step", "exit"};
  for (auto command : commands)
  {
    visit_handle commandHandle;
    VisIt_CommandMetaData_alloc(&commandHandle);
    VisIt_CommandMetaData_setName(commandHandle, command);
    VisIt_SimulationMetaData_addGenericCommand(simHandle, commandHandle);
  }

  visit_handle meshHandle;
  VisIt_MeshMetaData_alloc(&meshHandle);
  VisIt_MeshMetaData_setName(meshHandle, "mesh");
  VisIt_MeshMetaData_setMeshType(meshHandle, VISIT_MESHTYPE_UNSTRUCTURED);
  VisIt_MeshMetaData_setTopologicalDimension(meshHandle, 3);
  VisIt_MeshMetaData_setNumDomains(meshHandle, bridge->Internals->NumBlocks);
  VisIt_MeshMetaData_setDomainTitle(meshHandle, "Domains");
  VisIt_MeshMetaData_setDomainPieceName(meshHandle, "domain");
  VisIt_SimulationMetaData_addMesh(simHandle, meshHandle);

  visit_handle dataHandle;
  VisIt_VariableMetaData_alloc(&dataHandle);
  VisIt_VariableMetaData_setName(dataHandle, "oscillator");
  VisIt_VariableMetaData_setMeshName(dataHandle, "mesh");
  VisIt_VariableMetaData_setCentering(dataHandle, VISIT_VARCENTERING_ZONE);
  VisIt_VariableMetaData_setType(dataHandle, VISIT_VARTYPE_SCALAR);
  VisIt_SimulationMetaData_addVariable(simHandle, dataHandle);
  return simHandle;
}

static size_t numVertices(const diy::DiscreteBounds &bounds)
{
  int nx = bounds.max[0] - bounds.min[0] + 1 + 1;
  int ny = bounds.max[1] - bounds.min[1] + 1 + 1;
  int nz = bounds.max[2] - bounds.min[2] + 1 + 1;
  return nx * ny * nz;
}

static size_t numElements(const diy::DiscreteBounds &bounds)
{
  int nx = bounds.max[0] - bounds.min[0] + 1;
  int ny = bounds.max[1] - bounds.min[1] + 1;
  int nz = bounds.max[2] - bounds.min[2] + 1;
  return nx * ny * nz;
}

static visit_handle newUnstructuredBlock(const double *origin,
                                         const double *spacing, const diy::DiscreteBounds &cellExts,
                                         bool structureOnly)
{
  // Add points.
  auto numVert = numVertices(cellExts);
  visit_handle meshHandle;
  std::array<visit_handle, 3> dataHandles;

  if (!VisIt_UnstructuredMesh_alloc(&meshHandle) == VISIT_OKAY)
    return VISIT_INVALID_HANDLE;
  for (size_t i = 0; i < 3; i++)
  {
    bridge->Internals->vertices[i].resize(numVert);
    VisIt_VariableData_alloc(&dataHandles[i]);
    VisIt_VariableData_setDataF(dataHandles[i], VISIT_OWNER_SIM, 1, numVert, bridge->Internals->vertices[i].data());
  }
  if (!structureOnly)
  {

    size_t idx = 0;

    for (int k = cellExts.min[2]; k <= cellExts.max[2] + 1; ++k)
    {
      double z = origin[2] + spacing[2] * k;
      for (int j = cellExts.min[1]; j <= cellExts.max[1] + 1; ++j)
      {
        double y = origin[1] + spacing[1] * j;
        for (int i = cellExts.min[0]; i <= cellExts.max[0] + 1; ++i)
        {
          double x = origin[0] + spacing[0] * i;
          bridge->Internals->vertices[0].at(idx) = x;
          bridge->Internals->vertices[1].at(idx) = y;
          bridge->Internals->vertices[2].at(idx) = z;
          ++idx;
        }
      }
    }
    VisIt_UnstructuredMesh_setCoordsXYZ(meshHandle, dataHandles[0], dataHandles[1], dataHandles[2]);

    // Add cells
    int nx = cellExts.max[0] - cellExts.min[0] + 1 + 1;
    int ny = cellExts.max[1] - cellExts.min[1] + 1 + 1;
    int nz = cellExts.max[2] - cellExts.min[2] + 1 + 1;
    int ncx = nx - 1;
    int ncy = ny - 1;
    int ncz = nz - 1;
    size_t ncells = ncx * ncy * ncz;

    int ghostWidth[3][2];

    for (unsigned i = 0; i < 3; i++)
    {
      ghostWidth[i][0] =
          cellExts.min[i] + bridge->Internals->NumGhostCells == bridge->Internals->DomainExtent.min[i] ? 0 : bridge->Internals->NumGhostCells;
      ghostWidth[i][1] =
          cellExts.max[i] - bridge->Internals->NumGhostCells == bridge->Internals->DomainExtent.max[i] ? 0 : bridge->Internals->NumGhostCells;
    }

    auto &cl = bridge->Internals->cl;
    cl.resize(ncells * 9);
    auto &ghostIndicees = bridge->Internals->ghostIndicees;
    ghostIndicees.reserve(ncells);
    idx = 0;
    int nxny = (ncx + 1) * (ncy + 1);
    int offset = 0;
    for (int k = 0; k < ncz; ++k)
      for (int j = 0; j < ncy; ++j)
        for (int i = 0; i < ncx; ++i)
        {
          cl[idx++] = VISIT_CELL_HEX;
          cl[idx++] = (k)*nxny + j * nx + i;
          cl[idx++] = (k + 1) * nxny + j * nx + i;
          cl[idx++] = (k + 1) * nxny + j * nx + i + 1;
          cl[idx++] = (k)*nxny + j * nx + i + 1;
          cl[idx++] = (k)*nxny + (j + 1) * nx + i;
          cl[idx++] = (k + 1) * nxny + (j + 1) * nx + i;
          cl[idx++] = (k + 1) * nxny + (j + 1) * nx + i + 1;
          cl[idx++] = (k)*nxny + (j + 1) * nx + i + 1;
          if ((i < ghostWidth[0][0] || i + ghostWidth[0][1] >= nx) ||
              (j < ghostWidth[1][0] || j + ghostWidth[1][1] >= ny) ||
              (k < ghostWidth[2][0] || k + ghostWidth[2][1] >= nz))
            ghostIndicees.push_back(true);
          else
            ghostIndicees.push_back(false);
        }

    visit_handle clHandle;
    VisIt_VariableData_alloc(&clHandle);
    VisIt_VariableData_setDataI(clHandle, VISIT_OWNER_SIM, 1, ncells * 9, cl.data());
    VisIt_UnstructuredMesh_setConnectivity(meshHandle, ncells, clHandle);

    visit_handle ghostHandle;
    VisIt_VariableData_alloc(&ghostHandle);
    VisIt_VariableData_setDataC(ghostHandle, VISIT_OWNER_SIM, 1, ncells, ghostIndicees.data());
    VisIt_UnstructuredMesh_setGhostCells(meshHandle, ghostHandle);
    assert(ncells *9 == cl.size());
    assert(ncells == ghostIndicees.size());
  }

  return meshHandle;
}

static visit_handle VisItGetMesh(int gid, const char *name, void *ptr)
{
  if (!strcmp(name, "mesh"))
  {
    auto it = bridge->Internals->BlockExtents.find(gid);
    if (it == bridge->Internals->BlockExtents.end())
      return VISIT_INVALID_HANDLE;
    return newUnstructuredBlock(bridge->Internals->Origin, bridge->Internals->Spacing, it->second, false);
  }
  else if (!strcmp(name, "ucdmesh"))
  {
  }
  return 0;
}

static visit_handle VisItGetVariable(int gid, const char *name, void *ptr)
{
  if (!strcmp(name, "oscillator"))
  {
    visit_handle dataHandle;
    VisIt_VariableData_alloc(&dataHandle);

    auto extents = bridge->Internals->BlockExtents.find(gid);
    auto data = bridge->Internals->BlockData.find(gid);
    if (extents == bridge->Internals->BlockExtents.end() || data == bridge->Internals->BlockData.end())
      return VISIT_INVALID_HANDLE;
    VisIt_VariableData_setDataF(dataHandle, VISIT_OWNER_SIM, 1, numElements(extents->second), data->second);
    return dataHandle;
  }
  return VISIT_INVALID_HANDLE;
}

static visit_handle VisItGetDomainList(const char *name, void *ptr)
{
  visit_handle domainHandle;
  VisIt_DomainList_alloc(&domainHandle);
  visit_handle dataHandle;
  VisIt_VariableData_alloc(&dataHandle);
  std::vector<int> myDoms;
  for (const auto &block : bridge->Internals->BlockExtents)
    myDoms.push_back(block.first);
  VisIt_VariableData_setDataI(dataHandle, VISIT_OWNER_COPY, 1, bridge->Internals->BlockExtents.size(), myDoms.data());
  VisIt_DomainList_setDomains(domainHandle, bridge->Internals->NumBlocks, dataHandle);
  return domainHandle;
}

static visit_handle VisItGetDomainBoundaries(const char *name, void *ptr)
{
  return VISIT_INVALID_HANDLE;
}

//-----------------------------------------------------------------------------

static void setVisItCallbacks()
{
  VisItSetSlaveProcessCallback(&VisItSlaveProcessCallback);
  VisItSetCommandCallback(&VisitCommandCallback, nullptr);
  VisItSetActivateTimestep(VisItActivateTimestep, NULL);
  VisItSetGetMetaData(VisItGetMetaData, NULL);
  VisItSetGetMesh(VisItGetMesh, NULL);
  VisItSetGetVariable(VisItGetVariable, NULL);
  VisItSetGetDomainList(VisItGetDomainList, NULL);
  VisItSetGetDomainBoundaries(VisItGetDomainBoundaries, NULL);
}

static int getVisitState(bool blocking)
{
  int visitState = 10;
  if (bridge->rank == 0)
  {
    visitState = VisItDetectInput(blocking, -1);
  }
  visit_broadcast_int_callback(&visitState, 0, nullptr);
  switch (visitState)
  {
  case 0:
  {
    if (blocking)
      std::cerr << "blocking call should not have timed out" << std::endl;
  }
  break;
  case 1:
  {
    if (!VisItAttemptToCompleteConnection())
      std::cerr << "VisItAttemptToCompleteConnection failed" << std::endl;
    else
      setVisItCallbacks();
  }
  break;
  case 2:
  {
    if (!VisItProcessEngineCommand())
      VisItDisconnect();
  }
  break;
  case 3:
    std::cerr << "console input detected" << std::endl;
    break;
  case -5:
  case -4:
  case -3:
  case -2:
  case -1:
    std::cerr << "Error in VisItDetectInput " << visitState << std::endl;
    break;
  default:
    break;
  }
  return visitState;
}

//-----------------------------------------------------------------------------
void SetBlockExtent(int gid, int xmin, int xmax, int ymin,
                    int ymax, int zmin, int zmax)
{
  bridge->Internals->BlockExtents.insert_or_assign(gid, diy::DiscreteBounds{diy::DiscreteBounds::Point{xmin, ymin, zmin},
                                                                            diy::DiscreteBounds::Point{xmax, ymax, zmax}});
}

//-----------------------------------------------------------------------------
void SetDomainExtent(int xmin, int xmax, int ymin,
                     int ymax, int zmin, int zmax)
{
  bridge->Internals->DomainExtent.min[0] = xmin;
  bridge->Internals->DomainExtent.min[1] = ymin;
  bridge->Internals->DomainExtent.min[2] = zmin;
  bridge->Internals->DomainExtent.max[0] = xmax;
  bridge->Internals->DomainExtent.max[1] = ymax;
  bridge->Internals->DomainExtent.max[2] = zmax;
}

Bridge::Bridge(const MPI_Comm &communicator)
    : comm(communicator)
{
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bridge = this;
}

int Bridge::initialize(size_t nblocks, size_t n_local_blocks,
                       float *origin, float *spacing, int domain_shape_x, int domain_shape_y,
                       int domain_shape_z, int *gid, int *from_x, int *from_y, int *from_z,
                       int *to_x, int *to_y, int *to_z, int *shape, int ghostLevels,
                       const std::string &config_file)
{
  Internals = new InternalsType{};
  VisItSetBroadcastIntFunction2(&visit_broadcast_int_callback, nullptr);
  VisItSetBroadcastStringFunction2(&visit_broadcast_string_callback, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  VisItSetParallel(size > 1);
  VisItSetParallelRank(rank);
  VisItSetDirectory((char *)VISIT_INSTALL_DIR);
  VisItSetOptions((char *)"");
  VisItSetupEnvironment();
  if (rank == 0)
  {
    VisItInitializeSocketAndDumpSimFile("MiniSim", "based on senseis oscillator example", std::experimental::filesystem::current_path().c_str(), nullptr, nullptr, nullptr);
  }
  int visitState = getVisitState(true);

  Internals->NumBlocks = nblocks;

  for (int i = 0; i < 3; ++i)
    Internals->Origin[i] = origin[i];

  for (int i = 0; i < 3; ++i)
    Internals->Spacing[i] = spacing[i];

  for (int i = 0; i < 3; ++i)
    Internals->Shape[i] = shape[i];

  Internals->NumGhostCells = ghostLevels;

  SetDomainExtent(0, domain_shape_x - 1, 0,
                  domain_shape_y - 1, 0, domain_shape_z - 1);

  for (size_t cc = 0; cc < n_local_blocks; ++cc)
  {
    SetBlockExtent(gid[cc],
                   from_x[cc], to_x[cc], from_y[cc], to_y[cc],
                   from_z[cc], to_z[cc]);
  }

  return 0;
}

//-----------------------------------------------------------------------------
void Bridge::set_data(int gid, float *data)
{
  Internals->BlockData[gid] = data;
  auto max = std::max_element(data, data + numElements(Internals->BlockExtents.find(gid)->second));
  auto min = std::min_element(data, data + numElements(Internals->BlockExtents.find(gid)->second));
  std::cerr << "max = " << *max << ", min = " << *min << " size" << numElements(Internals->BlockExtents.find(gid)->second) << std::endl;
}

//-----------------------------------------------------------------------------
void Bridge::set_particles(int gid, const std::vector<Particle> &particles)
{
  Internals->ParticleData[gid] = &particles;
}

//-----------------------------------------------------------------------------
bool Bridge::execute(long step, float time)
{
  Internals->time = time;
  Internals->cycle = step;
  if (VisItIsConnected())
  {
    if (runflag || doAStep)
    {
      getVisitState(false);
    }
    else
    {
      while (!runflag && !doAStep && !terminate)
      {
        int visitState = getVisitState(true);
      }
    }
    doAStep = false;
    if (terminate)
      return false;
    VisItTimeStepChanged();
  }
  return true;
}

//-----------------------------------------------------------------------------
void Bridge::finalize()
{
  VisItDisconnect();
  delete Internals;
}
