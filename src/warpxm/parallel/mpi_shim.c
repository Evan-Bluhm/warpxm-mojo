/* MPI shim for warpxm-mojo.
 *
 * Mojo's `external_call` can call C functions but cannot name C globals,
 * and MPI handles (MPI_COMM_WORLD, MPI_DOUBLE, MPI_REQUEST_NULL) are
 * globals in Open MPI and integers in MPICH. This file keeps every handle
 * on the C side and exposes plain scalar functions, so `parallel/mpi.mojo`
 * is independent of the MPI implementation.
 *
 * Lifecycle is implicit: the first call initialises MPI and registers an
 * exit hook. A driver never calls init or finalize.
 *
 * Build (see scripts/mpi_build.sh):
 *     mpicc -O2 -fPIC -c src/warpxm/parallel/mpi_shim.c -o build/mpi_shim.o
 */

#define _DEFAULT_SOURCE
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#if defined(__has_include)
#  if __has_include(<mpi-ext.h>)
#    include <mpi-ext.h> /* MPIX_Query_cuda_support / _rocm_support */
#  endif
#endif

/* --- lifecycle -------------------------------------------------------- */

#if defined(__GLIBC__)
/* A rank that exits with an error (an uncaught Mojo `raise`) must take the
 * job down; a plain MPI_Finalize there leaves its neighbours blocked in
 * MPI_Waitall. `on_exit` is the glibc hook that sees the exit status. */
static void wxm_on_exit(int status, void *unused) {
    (void)unused;
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) return;
    if (status != 0) MPI_Abort(MPI_COMM_WORLD, status);
    MPI_Finalize();
}
#else
static void wxm_at_exit(void) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();
}
#endif

static void wxm_ensure_init(void) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) return;
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
#if defined(__GLIBC__)
    on_exit(wxm_on_exit, NULL);
#else
    atexit(wxm_at_exit);
#endif
}

int wxm_mpi_rank(void) {
    wxm_ensure_init();
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

int wxm_mpi_size(void) {
    wxm_ensure_init();
    int s = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
}

/* Index of this rank among the ranks on the same node: the device a rank
 * should use when a node's GPUs are all visible to every rank. */
int wxm_mpi_node_rank(void) {
    wxm_ensure_init();
    MPI_Comm node;
    int r = 0;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &r);
    MPI_Comm_free(&node);
    return r;
}

void wxm_mpi_abort(int code) {
    wxm_ensure_init();
    MPI_Abort(MPI_COMM_WORLD, code);
}

double wxm_mpi_wtime(void) {
    wxm_ensure_init();
    return MPI_Wtime();
}

void wxm_mpi_barrier(void) {
    wxm_ensure_init();
    MPI_Barrier(MPI_COMM_WORLD);
}

/* 1 if MPI_Isend / MPI_Irecv accept device pointers. The environment
 * variable WXM_GPU_AWARE_MPI=0|1 overrides the library's own answer (some
 * builds are GPU-aware through UCX without advertising it, and the
 * reverse). */
int wxm_mpi_gpu_aware(void) {
    wxm_ensure_init();
    const char *env = getenv("WXM_GPU_AWARE_MPI");
    if (env != NULL && env[0] != '\0') return env[0] != '0';
    int aware = 0;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    aware = aware || MPIX_Query_cuda_support();
#endif
#if defined(MPIX_ROCM_AWARE_SUPPORT) && MPIX_ROCM_AWARE_SUPPORT
    aware = aware || MPIX_Query_rocm_support();
#endif
    return aware;
}

/* --- reductions over all ranks --------------------------------------- */

double wxm_mpi_sum(double x) {
    wxm_ensure_init();
    double out = x;
    MPI_Allreduce(&x, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

double wxm_mpi_min(double x) {
    wxm_ensure_init();
    double out = x;
    MPI_Allreduce(&x, &out, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return out;
}

double wxm_mpi_max(double x) {
    wxm_ensure_init();
    double out = x;
    MPI_Allreduce(&x, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return out;
}

unsigned long long wxm_mpi_sum_u64(unsigned long long x) {
    wxm_ensure_init();
    unsigned long long out = x;
    MPI_Allreduce(&x, &out, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    return out;
}

/* --- halo exchange ----------------------------------------------------
 *
 * One call per exchange. `send` and `recv` are laid out as up to six
 * consecutive blocks, one per direction d = 2*axis + side, of
 * `count[d] * stride` doubles each. Block d of `send` goes to rank
 * `neighbor[d]`; block d of `recv` comes from it. A direction with
 * `neighbor[d] < 0` or `count[d] == 0` is skipped and occupies no space.
 *
 * Tags carry the sender's direction, so a rank that is both the -x and the
 * +x neighbour (two ranks along a periodic axis) keeps the two messages
 * apart: what we send towards d is what the neighbour receives from d^1.
 */
int wxm_mpi_halo_exchange(const double *send, double *recv, int stride,
                          int n0, int n1, int n2, int n3, int n4, int n5,
                          int c0, int c1, int c2, int c3, int c4, int c5) {
    wxm_ensure_init();
    const int neighbor[6] = {n0, n1, n2, n3, n4, n5};
    const int count[6] = {c0, c1, c2, c3, c4, c5};
    MPI_Request req[12];
    int nreq = 0;
    long offset = 0;
    for (int d = 0; d < 6; d++) {
        if (neighbor[d] < 0 || count[d] == 0) continue;
        int len = count[d] * stride;
        MPI_Irecv(recv + offset, len, MPI_DOUBLE, neighbor[d], d ^ 1,
                  MPI_COMM_WORLD, &req[nreq++]);
        offset += len;
    }
    offset = 0;
    for (int d = 0; d < 6; d++) {
        if (neighbor[d] < 0 || count[d] == 0) continue;
        int len = count[d] * stride;
        MPI_Isend(send + offset, len, MPI_DOUBLE, neighbor[d], d,
                  MPI_COMM_WORLD, &req[nreq++]);
        offset += len;
    }
    return MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
}
