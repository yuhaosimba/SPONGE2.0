#pragma once

#include "Selective_Policies.h"

template <class Policy, bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Selective_LJ_Direct_Coulomb_Device(
    const int local_atom_numbers, const int solvent_numbers,
    const ATOM_GROUP* nl, const VECTOR_LJ* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int* atom_sys_mark, const float* LJ_type_A,
    const float* LJ_type_B, const float cutoff, const float pme_beta,
    const Policy policy)
{
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < local_atom_numbers - solvent_numbers)
#else
#pragma omp parallel for schedule(dynamic)
    for (int atom_i = 0; atom_i < local_atom_numbers - solvent_numbers;
         atom_i++)
#endif
    {
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR_LJ r1 = crd[atom_i];
        SELECTIVE_PAIR_ACCUMULATOR acc;
        int atom_mark_i = atom_sys_mark[atom_i];
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < local_atom_numbers ? 1.0f : 0.5f;
            VECTOR_LJ r2 = crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int atom_pair_LJ_type = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                float pair_lj = 0.0f;
                float pair_coulomb = 0.0f;
                float frc_abs = 0.0f;
                float A = LJ_type_A[atom_pair_LJ_type];
                float B = LJ_type_B[atom_pair_LJ_type];
                int mark_sum = atom_mark_i + atom_sys_mark[atom_j];

                if (need_force)
                {
                    frc_abs = Get_LJ_Force(r1, r2, dr_abs, A, B);
                    if (need_coulomb)
                    {
                        float frc_cf_abs =
                            Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
                        frc_abs = frc_abs - frc_cf_abs;
                    }
                }
                if (need_energy)
                {
                    pair_lj = Get_LJ_Energy(r1, r2, dr_abs, A, B);
                    if (need_coulomb)
                    {
                        pair_coulomb =
                            Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    }
                }
                policy.template Accumulate_Pair<need_force, need_energy,
                                                need_virial, need_coulomb>(
                    atom_j, local_atom_numbers, ij_factor, dr, frc_abs, pair_lj,
                    pair_coulomb, mark_sum, &acc);
            }
        }
        policy.template Write_Atom<need_force, need_energy, need_virial,
                                   need_coulomb>(atom_i, &acc);
    }
}

template <class Policy, bool need_force, bool need_energy, bool need_virial,
          bool need_coulomb>
static __global__ void Selective_LJ_Direct_Coulomb_Soft_Core_Device(
    const int local_atom_numbers, const int solvent_numbers,
    const ATOM_GROUP* nl, const VECTOR_LJ_SOFT_TYPE* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const int* atom_sys_mark, const float* LJ_type_AA,
    const float* LJ_type_AB, const float* LJ_type_BA, const float* LJ_type_BB,
    const float cutoff, const float pme_beta, const float lambda,
    const float alpha, const float p, const float input_sigma_6,
    const float input_sigma_6_min, const Policy policy)
{
    float lambda_ = 1.0 - lambda;
#ifdef USE_GPU
    int atom_i = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom_i < local_atom_numbers - solvent_numbers)
#else
#pragma omp parallel for schedule(dynamic)
    for (int atom_i = 0; atom_i < local_atom_numbers - solvent_numbers;
         atom_i++)
#endif
    {
        ATOM_GROUP nl_i = nl[atom_i];
        VECTOR_LJ_SOFT_TYPE r1 = crd[atom_i];
        SELECTIVE_PAIR_ACCUMULATOR acc;
        int atom_mark_i = atom_sys_mark[atom_i];
#ifdef USE_GPU
        for (int j = threadIdx.x; j < nl_i.atom_numbers; j += blockDim.x)
#else
        for (int j = 0; j < nl_i.atom_numbers; j++)
#endif
        {
            int atom_j = nl_i.atom_serial[j];
            float ij_factor = atom_j < local_atom_numbers ? 1.0f : 0.5f;
            VECTOR_LJ_SOFT_TYPE r2 = crd[atom_j];
            VECTOR dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
            float dr_abs = norm3df(dr.x, dr.y, dr.z);
            if (dr_abs < cutoff)
            {
                int mark_sum = atom_mark_i + atom_sys_mark[atom_j];
                int atom_pair_LJ_type_A = Get_LJ_Type(r1.LJ_type, r2.LJ_type);
                int atom_pair_LJ_type_B =
                    Get_LJ_Type(r1.LJ_type_B, r2.LJ_type_B);
                float AA = LJ_type_AA[atom_pair_LJ_type_A];
                float AB = LJ_type_AB[atom_pair_LJ_type_A];
                float BA = LJ_type_BA[atom_pair_LJ_type_B];
                float BB = LJ_type_BB[atom_pair_LJ_type_B];

                float pair_lj = 0.0f;
                float pair_coulomb = 0.0f;
                float frc_abs = 0.0f;
                if (BA * AA != 0 || BA + AA == 0)
                {
                    if (need_energy)
                    {
                        pair_lj =
                            lambda_ * Get_LJ_Energy(r1, r2, dr_abs, AA, AB) +
                            lambda * Get_LJ_Energy(r1, r2, dr_abs, BA, BB);
                    }
                    if (need_force)
                    {
                        frc_abs =
                            lambda_ * Get_LJ_Force(r1, r2, dr_abs, AA, AB) +
                            lambda * Get_LJ_Force(r1, r2, dr_abs, BA, BB);
                    }
                    if (need_coulomb && need_energy)
                    {
                        pair_coulomb =
                            Get_Direct_Coulomb_Energy(r1, r2, dr_abs, pme_beta);
                    }
                    if (need_coulomb && need_force)
                    {
                        float frc_cf_abs =
                            Get_Direct_Coulomb_Force(r1, r2, dr_abs, pme_beta);
                        frc_abs = frc_abs - frc_cf_abs;
                    }
                }
                else
                {
                    float sigma_A = Get_Soft_Core_Sigma(AA, AB, input_sigma_6,
                                                        input_sigma_6_min);
                    float sigma_B = Get_Soft_Core_Sigma(BA, BB, input_sigma_6,
                                                        input_sigma_6_min);
                    float dr_softcore_A = Get_Soft_Core_Distance(
                        AA, AB, sigma_A, dr_abs, alpha, p, lambda);
                    float dr_softcore_B = Get_Soft_Core_Distance(
                        BB, BA, sigma_B, dr_abs, alpha, p, 1.0f - lambda);
                    if (need_energy)
                    {
                        pair_lj = lambda_ * Get_LJ_Energy(r1, r2, dr_softcore_A,
                                                          AA, AB) +
                                  lambda * Get_LJ_Energy(r1, r2, dr_softcore_B,
                                                         BA, BB);
                    }
                    if (need_force)
                    {
                        frc_abs =
                            lambda_ * Get_Soft_Core_LJ_Force(r1, r2, dr_abs,
                                                             dr_softcore_A, AA,
                                                             AB) +
                            lambda * Get_Soft_Core_LJ_Force(
                                         r1, r2, dr_abs, dr_softcore_B, BA, BB);
                    }
                    if (need_coulomb && need_energy)
                    {
                        pair_coulomb =
                            lambda_ * Get_Direct_Coulomb_Energy(
                                          r1, r2, dr_softcore_A, pme_beta) +
                            lambda * Get_Direct_Coulomb_Energy(
                                         r1, r2, dr_softcore_B, pme_beta);
                    }
                    if (need_coulomb && need_force)
                    {
                        float frc_cf_abs =
                            lambda_ *
                                Get_Soft_Core_Direct_Coulomb_Force(
                                    r1, r2, dr_abs, dr_softcore_A, pme_beta) +
                            lambda *
                                Get_Soft_Core_Direct_Coulomb_Force(
                                    r1, r2, dr_abs, dr_softcore_B, pme_beta);
                        frc_abs = frc_abs - frc_cf_abs;
                    }
                }
                policy.template Accumulate_Pair<need_force, need_energy,
                                                need_virial, need_coulomb>(
                    atom_j, local_atom_numbers, ij_factor, dr, frc_abs, pair_lj,
                    pair_coulomb, mark_sum, &acc);
            }
        }
        policy.template Write_Atom<need_force, need_energy, need_virial,
                                   need_coulomb>(atom_i, &acc);
    }
}
