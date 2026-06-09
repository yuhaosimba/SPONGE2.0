#include "PM_force.h"

__global__ void charge_square_kernel(int element_number, const float* charge,
                                     float* charge_square);
__global__ void PME_Add_Energy_To_Potential(float* d_ene, float* d_self_ene,
                                            float* d_reciprocal_ene);
__global__ void PME_Sum_Virial(const int nfft, const LTMatrix3* virial_BC,
                               const FFT_COMPLEX* FQ, LTMatrix3* virial,
                               int fftz);
__global__ void device_add_force(const int atom_numbers, float update_interval,
                                 VECTOR* force, const VECTOR* force_backup);
__global__ void PME_BCFQ(FFT_COMPLEX* PME_FQ, float* PME_BC, int PME_Nfft);
__global__ void ESP_Atom_Near(const VECTOR* crd, const LTMatrix3 cell,
                              const LTMatrix3 rcell, const int atom_numbers,
                              const int fftx, const int ffty, const int fftz,
                              UNSIGNED_INT_VECTOR* PME_uxyz, VECTOR* PME_frxyz,
                              VECTOR* force_backup);
__global__ void ESP_Q_Spread_Order5(
    const UNSIGNED_INT_VECTOR* PME_uxyz, const float* charge,
    const VECTOR* PME_frxyz, float* PME_Q, const int atom_numbers,
    const int PME_Nin, const int PME_Nall, const int fftx, const int ffty,
    const int fftz, const int table_points, const int poly_order,
    const int use_poly, const float* window_table, const float* window_coeff);
__global__ void ESP_Q_Spread(
    const UNSIGNED_INT_VECTOR* PME_uxyz, const float* charge,
    const VECTOR* PME_frxyz, float* PME_Q, const int atom_numbers,
    const int PME_Nin, const int PME_Nall, const int fftx, const int ffty,
    const int fftz, const int order, const int support, const int table_points,
    const int poly_order, const int use_poly, const float* window_table,
    const float* window_coeff);
__global__ void ESP_Final_Order5(
    const UNSIGNED_INT_VECTOR* PME_uxyz, const float* charge, const float* PME_Q,
    VECTOR* force, const VECTOR* PME_frxyz, const LTMatrix3 rcell,
    const int fftx, const int ffty, const int fftz, const int atom_numbers,
    const int PME_Nin, const int PME_Nall, const int table_points,
    const int poly_order, const int use_poly, const float* window_table,
    const float* window_derivative_table, const float* window_coeff,
    const float* window_derivative_coeff);
__global__ void ESP_Final(
    const UNSIGNED_INT_VECTOR* PME_uxyz, const float* charge, const float* PME_Q,
    VECTOR* force, const VECTOR* PME_frxyz, const LTMatrix3 rcell,
    const int fftx, const int ffty, const int fftz, const int atom_numbers,
    const int PME_Nin, const int PME_Nall, const int order, const int support,
    const int table_points, const int poly_order, const int use_poly,
    const float* window_table, const float* window_derivative_table,
    const float* window_coeff, const float* window_derivative_coeff);

static constexpr int ESP_ORDER5 = 5;
static constexpr int ESP_ORDER5_SUPPORT = 125;
static constexpr int ESP_GPU_SPREAD_LANES_PER_ATOM = 32;
static constexpr int ESP_GPU_SPREAD_ATOMS_PER_BLOCK = 4;
static constexpr int ESP_GPU_FINAL_LANES_PER_ATOM = 8;
static constexpr int ESP_GPU_FINAL_ATOMS_PER_BLOCK = 128;

static __global__ void ESP_Energy_Product(const int element_number,
                                          const float* list1,
                                          const float* list2, float* sum)
{
#ifdef USE_GPU
    if (threadIdx.x == 0)
    {
        sum[0] = 0.;
    }
    __syncthreads();
#else
    sum[0] = 0;
#endif
    double lin = 0.0;
#ifdef USE_GPU
    for (int i = threadIdx.x; i < element_number; i = i + blockDim.x)
#else
#pragma omp parallel for reduction(+ : lin)
    for (int i = 0; i < element_number; i++)
#endif
    {
        double product = 0.0;
        if (PM_Float_Is_Bounded(list1[i], 1.0e6f) &&
            PM_Float_Is_Bounded(list2[i], 1.0e6f))
        {
            product = static_cast<double>(list1[i]) * list2[i];
            if (product < 1.0e12 && product > -1.0e12)
            {
                lin += product;
            }
        }
    }
    if (lin > 1.0e20 || lin < -1.0e20)
    {
        lin = 0.0;
    }
    atomicAdd(sum, static_cast<float>(lin));
}

static __global__ void up_box_esp_bc(
    int fftx, int ffty, int fftz, float* ESP_BC, const float* ESP_BC0,
    LTMatrix3* ESP_virial_BC, LTMatrix3 rcell, float volume, float cutoff,
    float c_split, int table_points, int split_poly_order, int use_poly,
    const float* split_fourier_table, const float* split_fourier_coeff,
    const float* split_fourier_derivative_table,
    const float* split_fourier_derivative_coeff)
{
    float kxrp, kyrp, kzrp;
    int ky, kz, index;
    float msq;
    VECTOR m;
    LTMatrix3 virial_bc_local;
    float bc_local;
    float split_fourier, split_fourier_derivative;
    float derivative_ratio, metric_factor;
    const float split_scale = 2.0f * CONSTANT_Pi * cutoff / c_split;
#ifdef USE_GPU
    for (int kx = blockIdx.x * blockDim.x + threadIdx.x; kx < fftx;
         kx += blockDim.x * gridDim.x)
#else
#pragma omp parallel for firstprivate(                                      \
        kxrp, kyrp, kzrp, ky, kz, index, msq, m, virial_bc_local, bc_local, \
            split_fourier, split_fourier_derivative, derivative_ratio,      \
            metric_factor)
    for (int kx = 0; kx < fftx; kx++)
#endif
    {
        kxrp = kx;
        if (kx > fftx / 2) kxrp = kx - fftx;
#ifdef USE_GPU
        for (ky = blockIdx.y * blockDim.y + threadIdx.y; ky < ffty;
             ky += blockDim.y * gridDim.y)
#else
        for (ky = 0; ky < ffty; ky++)
#endif
        {
            kyrp = ky;
            if (ky > ffty / 2) kyrp = ky - ffty;
#ifdef USE_GPU
            for (kz = threadIdx.z; kz <= fftz / 2; kz += blockDim.z)
#else
            for (kz = 0; kz <= fftz / 2; kz++)
#endif
            {
                kzrp = kz;
                m = {kxrp, kyrp, kzrp};
                m = MultiplyTranspose(m, rcell);
                msq = m * m;

                index = kx * ffty * (fftz / 2 + 1) + ky * (fftz / 2 + 1) + kz;

                ESP_BC[index] = 0.0f;
                ESP_virial_BC[index] = {0, 0, 0, 0, 0, 0};
                if (kx + ky + kz == 0 || msq <= 0.0f)
                {
                    continue;
                }

                float split_arg = split_scale * sqrtf(msq);
                if (split_arg > 1.0f)
                {
                    continue;
                }

                split_fourier =
                    0.5f * PM_Eval_Direct_Scalar(
                               split_fourier_table, split_fourier_coeff,
                               table_points, split_poly_order, use_poly,
                               split_arg);
                split_fourier_derivative =
                    0.5f * PM_Eval_Direct_Scalar(
                               split_fourier_derivative_table,
                               split_fourier_derivative_coeff, table_points,
                               split_poly_order, use_poly, split_arg);
                if (!PM_Float_Is_Bounded(ESP_BC0[index], 1.0e12f))
                {
                    continue;
                }
                bc_local = split_fourier * ESP_BC0[index] /
                           (CONSTANT_Pi * msq * volume);
                if (!PM_Float_Is_Bounded(bc_local, 1.0e6f))
                {
                    ESP_BC[index] = 0.0f;
                    continue;
                }
                ESP_BC[index] = bc_local;

                derivative_ratio = 0.0f;
                if (fabsf(split_fourier) > 1.0e-30f)
                {
                    derivative_ratio = split_fourier_derivative / split_fourier;
                }
                metric_factor = (2.0f - derivative_ratio) / msq;
                virial_bc_local.a11 = 1.0f - metric_factor * m.x * m.x;
                virial_bc_local.a21 = 0.0f - metric_factor * m.y * m.x;
                virial_bc_local.a22 = 1.0f - metric_factor * m.y * m.y;
                virial_bc_local.a31 = 0.0f - metric_factor * m.z * m.x;
                virial_bc_local.a32 = 0.0f - metric_factor * m.z * m.y;
                virial_bc_local.a33 = 1.0f - metric_factor * m.z * m.z;
                ESP_virial_BC[index] = 0.5f * bc_local * virial_bc_local;
            }
        }
    }
}

PM_Direct_Parameters Particle_Mesh::Get_PM_Direct_Parameters() const
{
    PM_Direct_Parameters direct;
    direct.backend = backend;
    direct.table_points = esp.table_points;
    direct.split_poly_order = esp.split_poly_order;
    direct.use_polynomial_tables = esp.table_mode == ESPTableMode::POLY;
    direct.cutoff = esp.cutoff;
    direct.pme_beta = beta;
    direct.split_real_table = ESP_split_real_table;
    direct.split_real_derivative_table = ESP_split_real_derivative_table;
    direct.split_real_coeff = ESP_split_real_coeff;
    direct.split_real_derivative_coeff = ESP_split_real_derivative_coeff;
    return direct;
}

void Particle_Mesh::Validate_Direct_Force_Path(bool uses_selective_direct,
                                               bool uses_soft_core_direct) const
{
    if (backend == ParticleMeshBackend::ESP &&
        (uses_selective_direct || uses_soft_core_direct))
    {
        CONTROLLER controller;
        controller.Throw_SPONGE_Error(
            spongeErrorNotImplemented, "SPONGE main force loop",
            "Reason:\n\tESP direct-space split is implemented for the normal "
            "hard-core LJ/Coulomb path first. Selective direct-LJ/Coulomb and "
            "soft-core/FEP Coulomb still need the ESP-compatible "
            "derivatives.");
    }
}

void Particle_Mesh::Sanitize_Force(VECTOR* force, int atom_numbers) const
{
    (void)force;
    (void)atom_numbers;
    return;
}

void Run_ESP_Reciprocal_Energy_Backend(Particle_Mesh* pm, const float* charge,
                                       float* d_potential)
{
    Launch_Device_Kernel(ESP_Energy_Product, 1, CONTROLLER::device_max_thread,
                         0, NULL, pm->PME_Nall, pm->PME_Q, pm->PME_FBCFQ,
                         pm->d_reciprocal_ene);
    Scale_List(pm->d_reciprocal_ene, 0.5f, 1);

    Launch_Device_Kernel(
        charge_square_kernel,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->atom_numbers, charge,
        pm->charge_square);
    Sum_Of_List(pm->charge_square, pm->d_self_ene, pm->atom_numbers);
    Scale_List(pm->d_self_ene, -pm->esp.self_energy_coeff, 1);

    Launch_Device_Kernel(PME_Add_Energy_To_Potential, 1, 1, 0, NULL,
                         d_potential, pm->d_self_ene, pm->d_reciprocal_ene);
}

void Run_ESP_Reciprocal_Force_Backend(
    Particle_Mesh* pm, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, VECTOR* force, int need_virial,
    LTMatrix3* d_virial, int step)
{
    if (step % pm->update_interval != 0)
    {
        return;
    }

    int use_poly = pm->esp.table_mode == ESPTableMode::POLY;
    deviceMemset(pm->PME_Q, 0, sizeof(float) * pm->PME_Nall);
    Launch_Device_Kernel(
        ESP_Atom_Near,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, crd, cell, rcell,
        pm->atom_numbers, pm->fftx, pm->ffty, pm->fftz, pm->PME_uxyz,
        pm->PME_frxyz, pm->force_backup);

    dim3 blockSize(1, 1, 1);
    if (pm->esp.order == ESP_ORDER5)
    {
        blockSize = {ESP_GPU_SPREAD_LANES_PER_ATOM *
                         ESP_GPU_SPREAD_ATOMS_PER_BLOCK,
                     1};
        size_t spread_shared_memory =
            ESP_GPU_SPREAD_ATOMS_PER_BLOCK * ESP_ORDER5 *
            (3 * sizeof(float) + 3 * sizeof(int));
        Launch_Device_Kernel(
            ESP_Q_Spread_Order5,
            (pm->atom_numbers + ESP_GPU_SPREAD_ATOMS_PER_BLOCK - 1) /
                ESP_GPU_SPREAD_ATOMS_PER_BLOCK,
            blockSize, spread_shared_memory, NULL, pm->PME_uxyz, charge,
            pm->PME_frxyz, pm->PME_Q, pm->atom_numbers, pm->PME_Nin,
            pm->PME_Nall, pm->fftx, pm->ffty, pm->fftz,
            pm->esp.table_points, pm->esp.spread_poly_order, use_poly,
            pm->ESP_window_table, pm->ESP_window_coeff);
    }
    else
    {
        int spread_threads_y = pm->ESP_near_grid_points;
        if (spread_threads_y > CONTROLLER::device_max_thread)
            spread_threads_y = CONTROLLER::device_max_thread;
        if (spread_threads_y < 1) spread_threads_y = 1;
        blockSize = {CONTROLLER::device_max_thread / spread_threads_y,
                     (unsigned int)spread_threads_y};
        if (blockSize.x < 1) blockSize.x = 1;
        size_t spread_shared_memory =
            sizeof(float) * blockSize.x * pm->esp.order * 3;
        Launch_Device_Kernel(
            ESP_Q_Spread, (pm->atom_numbers + blockSize.x - 1) / blockSize.x,
            blockSize, spread_shared_memory, NULL, pm->PME_uxyz, charge,
            pm->PME_frxyz, pm->PME_Q, pm->atom_numbers, pm->PME_Nin,
            pm->PME_Nall, pm->fftx, pm->ffty, pm->fftz, pm->esp.order,
            pm->ESP_near_grid_points, pm->esp.table_points,
            pm->esp.spread_poly_order, use_poly, pm->ESP_window_table,
            pm->ESP_window_coeff);
    }
    SPONGE_FFT_WRAPPER::R2C(pm->PME_plan_r2c, pm->PME_Q, pm->PME_FQ);

    blockSize = {CONTROLLER::device_warp,
                 CONTROLLER::device_max_thread / CONTROLLER::device_warp};
    if (need_virial)
    {
        Launch_Device_Kernel(
            PME_Sum_Virial,
            (pm->PME_Nfft + 4 * CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            blockSize, 0, NULL, pm->PME_Nfft, pm->ESP_Virial_BC, pm->PME_FQ,
            d_virial, pm->fftz);
    }

    Launch_Device_Kernel(
        PME_BCFQ,
        (pm->PME_Nfft + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->PME_FQ, pm->ESP_BC,
        pm->PME_Nfft);

    SPONGE_FFT_WRAPPER::C2R(pm->PME_plan_c2r, pm->PME_FQ, pm->PME_FBCFQ);

    if (pm->esp.order == ESP_ORDER5)
    {
        blockSize = {ESP_GPU_FINAL_LANES_PER_ATOM *
                         ESP_GPU_FINAL_ATOMS_PER_BLOCK,
                     1};
        size_t final_shared_memory =
            ESP_GPU_FINAL_ATOMS_PER_BLOCK * ESP_ORDER5 *
            (6 * sizeof(float) + 3 * sizeof(int));
        Launch_Device_Kernel(
            ESP_Final_Order5,
            (pm->atom_numbers + ESP_GPU_FINAL_ATOMS_PER_BLOCK - 1) /
                ESP_GPU_FINAL_ATOMS_PER_BLOCK,
            blockSize, final_shared_memory, NULL, pm->PME_uxyz, charge,
            pm->PME_FBCFQ, pm->force_backup, pm->PME_frxyz, rcell, pm->fftx,
            pm->ffty, pm->fftz, pm->atom_numbers, pm->PME_Nin, pm->PME_Nall,
            pm->esp.table_points, pm->esp.spread_poly_order, use_poly,
            pm->ESP_window_table, pm->ESP_window_derivative_table,
            pm->ESP_window_coeff, pm->ESP_window_derivative_coeff);
    }
    else
    {
        blockSize = {8, CONTROLLER::device_max_thread / 8};
        size_t final_shared_memory =
            sizeof(float) * blockSize.y * pm->esp.order * 6;
        Launch_Device_Kernel(
            ESP_Final, (pm->atom_numbers + blockSize.y - 1) / blockSize.y,
            blockSize, final_shared_memory, NULL, pm->PME_uxyz, charge,
            pm->PME_FBCFQ, pm->force_backup, pm->PME_frxyz, rcell, pm->fftx,
            pm->ffty, pm->fftz, pm->atom_numbers, pm->PME_Nin, pm->PME_Nall,
            pm->esp.order, pm->ESP_near_grid_points, pm->esp.table_points,
            pm->esp.spread_poly_order, use_poly, pm->ESP_window_table,
            pm->ESP_window_derivative_table, pm->ESP_window_coeff,
            pm->ESP_window_derivative_coeff);
    }
    Launch_Device_Kernel(
        device_add_force,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->atom_numbers,
        pm->update_interval, force, pm->force_backup);
}

void Update_ESP_Box_Backend(Particle_Mesh* pm, LTMatrix3 rcell, float volume)
{
    dim3 blockSize = {8, 8, CONTROLLER::device_max_thread / 64};
    dim3 gridSize = {64, 64};
    pm->neutralizing_factor = 0.0f;
    Launch_Device_Kernel(
        up_box_esp_bc, gridSize, blockSize, 0, NULL, pm->fftx, pm->ffty,
        pm->fftz, pm->ESP_BC, pm->ESP_BC0, pm->ESP_Virial_BC, rcell, volume,
        pm->esp.cutoff, pm->esp.c_split, pm->esp.table_points,
        pm->esp.split_poly_order, pm->esp.table_mode == ESPTableMode::POLY,
        pm->ESP_split_fourier_table, pm->ESP_split_fourier_coeff,
        pm->ESP_split_fourier_derivative_table,
        pm->ESP_split_fourier_derivative_coeff);
}
