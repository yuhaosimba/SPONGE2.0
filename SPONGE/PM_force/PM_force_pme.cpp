#include "PM_force.h"

__global__ void charge_square_kernel(int element_number, const float* charge,
                                     float* charge_square);
__global__ void PME_Add_Energy_To_Potential(float* d_ene, float* d_self_ene,
                                            float* d_reciprocal_ene);
__global__ void PME_Atom_Near(const VECTOR* crd, int* PME_atom_near,
                              const int PME_Nin, const LTMatrix3 cell,
                              const LTMatrix3 rcell, const int atom_numbers,
                              const int fftx, const int ffty, const int fftz,
                              UNSIGNED_INT_VECTOR* PME_uxyz, VECTOR* PME_frxyz,
                              VECTOR* force_backup);
__global__ void PME_Q_Spread(int* PME_atom_near, const float* charge,
                             const VECTOR* PME_frxyz, float* PME_Q,
                             const int atom_numbers, const int PME_Nall);
__global__ void PME_BCFQ(FFT_COMPLEX* PME_FQ, float* PME_BC, int PME_Nfft);
__global__ void PME_Energy_Product(const int element_number, const float* list1,
                                   const float* list2, float* sum);
__global__ void device_add(float* ene, float factor, float* charge_sum);
__global__ void PME_Final(int* PME_atom_near, const float* charge,
                          const float* PME_Q, VECTOR* force,
                          const VECTOR* PME_frxyz, const LTMatrix3 rcell,
                          const int fftx, const int ffty, const int fftz,
                          const int atom_numbers, const int PME_Nall);
__global__ void PME_Sum_Virial(const int nfft, const LTMatrix3* virial_BC,
                               const FFT_COMPLEX* FQ, LTMatrix3* virial,
                               int fftz);
__global__ void device_add_force(const int atom_numbers, float update_interval,
                                 VECTOR* force, const VECTOR* force_backup);
__global__ void up_box_bc(int fftx, int ffty, int fftz, float* PME_BC,
                          float* PME_BC0, LTMatrix3* PME_virial_BC,
                          float mprefactor, LTMatrix3 rcell, float volume);

void Run_PME_Reciprocal_Force_Backend(
    Particle_Mesh* pm, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, VECTOR* force, int need_virial,
    LTMatrix3* d_virial, int step)
{
    if (step % pm->update_interval != 0)
    {
        return;
    }

    deviceMemset(pm->PME_Q, 0, sizeof(float) * pm->PME_Nall);
    Launch_Device_Kernel(
        PME_Atom_Near,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, crd, pm->PME_atom_near,
        pm->PME_Nin, cell, rcell, pm->atom_numbers, pm->fftx, pm->ffty,
        pm->fftz, pm->PME_uxyz, pm->PME_frxyz, pm->force_backup);

    dim3 blockSize = {CONTROLLER::device_max_thread / 64, 64};
    Launch_Device_Kernel(PME_Q_Spread,
                         (pm->atom_numbers + blockSize.x - 1) / blockSize.x,
                         blockSize, 0, NULL, pm->PME_atom_near, charge,
                         pm->PME_frxyz, pm->PME_Q, pm->atom_numbers,
                         pm->PME_Nall);

    SPONGE_FFT_WRAPPER::R2C(pm->PME_plan_r2c, pm->PME_Q, pm->PME_FQ);

    blockSize = {CONTROLLER::device_warp,
                 CONTROLLER::device_max_thread / CONTROLLER::device_warp};
    if (need_virial)
    {
        Launch_Device_Kernel(
            PME_Sum_Virial,
            (pm->PME_Nfft + 4 * CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            blockSize, 0, NULL, pm->PME_Nfft, pm->PME_Virial_BC, pm->PME_FQ,
            d_virial, pm->fftz);
    }

    Launch_Device_Kernel(
        PME_BCFQ,
        (pm->PME_Nfft + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->PME_FQ, pm->PME_BC,
        pm->PME_Nfft);

    SPONGE_FFT_WRAPPER::C2R(pm->PME_plan_c2r, pm->PME_FQ, pm->PME_FBCFQ);

    blockSize = {8, CONTROLLER::device_max_thread / 8};
    Launch_Device_Kernel(PME_Final,
                         (pm->atom_numbers + blockSize.x - 1) / blockSize.x,
                         blockSize, 0, NULL, pm->PME_atom_near, charge,
                         pm->PME_FBCFQ, pm->force_backup, pm->PME_frxyz, rcell,
                         pm->fftx, pm->ffty, pm->fftz, pm->atom_numbers,
                         pm->PME_Nall);

    Launch_Device_Kernel(
        device_add_force,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->atom_numbers,
        pm->update_interval, force, pm->force_backup);
}

void Run_PME_Reciprocal_Energy_Backend(Particle_Mesh* pm, const float* charge,
                                       float* d_potential)
{
    Launch_Device_Kernel(PME_Energy_Product, 1, CONTROLLER::device_max_thread,
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
    Scale_List(pm->d_self_ene, -pm->beta / sqrt(CONSTANT_Pi), 1);

    Sum_Of_List(charge, pm->charge_sum, pm->atom_numbers);
    Launch_Device_Kernel(device_add, 1, 1, 0, NULL, pm->d_self_ene,
                         pm->neutralizing_factor, pm->charge_sum);
    Launch_Device_Kernel(PME_Add_Energy_To_Potential, 1, 1, 0, NULL,
                         d_potential, pm->d_self_ene, pm->d_reciprocal_ene);
}

void Update_PME_Box_Backend(Particle_Mesh* pm, LTMatrix3 rcell, float volume)
{
    dim3 blockSize = {8, 8, CONTROLLER::device_max_thread / 64};
    dim3 gridSize = {64, 64};
    pm->neutralizing_factor =
        -0.5 * CONSTANT_Pi / (pm->beta * pm->beta * volume);
    float mprefactor =
        CONSTANT_Pi * CONSTANT_Pi / -pm->beta / pm->beta;
    Launch_Device_Kernel(up_box_bc, gridSize, blockSize, 0, NULL, pm->fftx,
                         pm->ffty, pm->fftz, pm->PME_BC, pm->PME_BC0,
                         pm->PME_Virial_BC, mprefactor, rcell, volume);
}
