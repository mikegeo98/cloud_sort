#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <cassert>

// global counters for memory transfers
static uint64_t total_host_to_device = 0;
static uint64_t total_device_to_host = 0;

// wrappers that count bytes transferred
#define ENQUEUE_WRITE(q, buf, block, off, sz, ptr, numEvents, eventList, event) \
    do { \
      total_host_to_device += (sz); \
      cl_int __err = clEnqueueWriteBuffer((q),(buf),(block),(off),(sz),(ptr),(numEvents),(eventList),(event)); \
      assert(__err == CL_SUCCESS); \
    } while(0)

#define ENQUEUE_READ(q, buf, block, off, sz, ptr, numEvents, eventList, event) \
    do { \
      total_device_to_host += (sz); \
      cl_int __err = clEnqueueReadBuffer((q),(buf),(block),(off),(sz),(ptr),(numEvents),(eventList),(event)); \
      assert(__err == CL_SUCCESS); \
    } while(0)

static std::string load(const char* path) {
  std::ifstream f(path);
  assert(f);
  return { std::istreambuf_iterator<char>(f), {} };
}

void run_opencl_radix(const std::vector<uint64_t>& in,
                      std::vector<uint64_t>&       out,
                      size_t N)
{
    const int    BITS      = 8;
    const int    RADIX     = 1 << BITS;
    const size_t LOCAL_SZ  = 256;
    const size_t NUM_GROUPS= (N + LOCAL_SZ - 1) / LOCAL_SZ;
    const size_t GLOBAL_SZ = NUM_GROUPS * LOCAL_SZ;
    const int    PASSES    = (64 + BITS - 1) / BITS;

    // Copy input to CPU baseline and sort for later verify
    std::vector<uint64_t> cpu = in;
    std::sort(cpu.begin(), cpu.end());
    out.resize(N);

    // 1) OpenCL init (platform, device, ctx, queue)
    cl_uint np; clGetPlatformIDs(0, nullptr, &np);
    std::vector<cl_platform_id> ps(np);
    clGetPlatformIDs(np, ps.data(), nullptr);
    cl_platform_id p = ps[0];

    cl_uint nd; clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
    std::vector<cl_device_id> ds(nd);
    clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, ds.data(), nullptr);
    cl_device_id d = ds[0];

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &d, nullptr, nullptr, &err);
    assert(err == CL_SUCCESS);
    cl_queue_properties props[] = { 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, d, props, &err);
    assert(err == CL_SUCCESS);

    // 2) Build program & kernels
    auto src = load("../kernels/radix_kernels.cl");
    const char* s = src.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &s, nullptr, &err);
    assert(err == CL_SUCCESS);
    err = clBuildProgram(prog, 1, &d, nullptr, nullptr, nullptr);
    assert(err == CL_SUCCESS);
    cl_kernel kh = clCreateKernel(prog, "build_group_histogram", &err);
    assert(err == CL_SUCCESS);
    cl_kernel ks = clCreateKernel(prog, "scatter_stable", &err);
    assert(err == CL_SUCCESS);

    // 3) Allocate buffers
    cl_mem buf_in  = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        N * sizeof(cl_ulong), (void*)in.data(), &err);
    assert(err == CL_SUCCESS);
    cl_mem buf_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                        N * sizeof(cl_ulong), nullptr, &err);
    assert(err == CL_SUCCESS);
    cl_mem buf_gh = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                        NUM_GROUPS * RADIX * sizeof(cl_uint), nullptr, &err);
    assert(err == CL_SUCCESS);
    cl_mem buf_pg = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                        RADIX * sizeof(cl_uint), nullptr, &err);
    assert(err == CL_SUCCESS);
    cl_mem buf_go = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                        NUM_GROUPS * RADIX * sizeof(cl_uint), nullptr, &err);
    assert(err == CL_SUCCESS);

    // temp host arrays
    std::vector<uint32_t> gh(NUM_GROUPS * RADIX),
                          bt(RADIX),
                          pg(RADIX),
                          go(NUM_GROUPS * RADIX),
                          zero(NUM_GROUPS * RADIX, 0);

    for (int pass = 0; pass < PASSES; ++pass) {
      uint32_t shift = pass * BITS;

      // zero per-group hist
      ENQUEUE_WRITE(q, buf_gh, CL_TRUE, 0, zero.size() * sizeof(zero[0]), zero.data(),
                    0, nullptr, nullptr);

      // build_group_histogram
      clSetKernelArg(kh, 0, sizeof(buf_in), &buf_in);
      clSetKernelArg(kh, 1, sizeof(buf_gh), &buf_gh);
      clSetKernelArg(kh, 2, sizeof(uint32_t), &N);
      clSetKernelArg(kh, 3, sizeof(uint32_t), &shift);
      clEnqueueNDRangeKernel(q, kh, 1, nullptr, &GLOBAL_SZ, &LOCAL_SZ, 0, nullptr, nullptr);
      clFinish(q);

      // read back gh
      ENQUEUE_READ(q, buf_gh, CL_TRUE, 0, gh.size() * sizeof(gh[0]), gh.data(),
                   0, nullptr, nullptr);

      // bucket_totals & prefix-sum across digits (host)
      for (int d = 0; d < RADIX; ++d) {
        uint32_t sum = 0;
        for (size_t g = 0; g < NUM_GROUPS; ++g)
          sum += gh[g * RADIX + d];
        bt[d] = sum;
      }
      pg[0] = 0;
      for (int d = 1; d < RADIX; ++d) pg[d] = pg[d - 1] + bt[d - 1];

      // group_offsets
      for (int d = 0; d < RADIX; ++d) {
        uint32_t sum = 0;
        for (size_t g = 0; g < NUM_GROUPS; ++g) {
          go[g * RADIX + d] = sum;
          sum += gh[g * RADIX + d];
        }
      }

      // upload pg & go
      ENQUEUE_WRITE(q, buf_pg, CL_TRUE, 0, pg.size() * sizeof(pg[0]), pg.data(),
                    0, nullptr, nullptr);
      ENQUEUE_WRITE(q, buf_go, CL_TRUE, 0, go.size() * sizeof(go[0]), go.data(),
                    0, nullptr, nullptr);

      // scatter_stable
      clSetKernelArg(ks, 0, sizeof(buf_in), &buf_in);
      clSetKernelArg(ks, 1, sizeof(buf_out), &buf_out);
      clSetKernelArg(ks, 2, sizeof(buf_pg), &buf_pg);
      clSetKernelArg(ks, 3, sizeof(buf_go), &buf_go);
      clSetKernelArg(ks, 4, sizeof(uint32_t), &N);
      clSetKernelArg(ks, 5, sizeof(uint32_t), &shift);
      clEnqueueNDRangeKernel(q, ks, 1, nullptr, &GLOBAL_SZ, &LOCAL_SZ, 0, nullptr, nullptr);
      clFinish(q);

      // swap buffers
      std::swap(buf_in, buf_out);
    }

    // read back final sorted data
    ENQUEUE_READ(q, buf_in, CL_TRUE, 0, N * sizeof(cl_ulong), out.data(),
                 0, nullptr, nullptr);

    // verify & report
    if (out != cpu) {
      std::cerr << "Mismatch!\n";
      std::exit(1);
    }

    std::cout << "PASS\n";
    std::cout << "Total H→D bytes: " << total_host_to_device << "\n";
    std::cout << "Total D→H bytes: " << total_device_to_host << "\n";

    // cleanup omitted for brevity...
}
