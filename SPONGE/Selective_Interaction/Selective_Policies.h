#pragma once

#include "../Lennard_Jones_force/LJ_soft_core.h"

struct SELECTIVE_PAIR_ACCUMULATOR
{
    VECTOR force = {0.0f, 0.0f, 0.0f};
    VECTOR selected_force = {0.0f, 0.0f, 0.0f};
    LTMatrix3 virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    LTMatrix3 selected_virial = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float energy_lj = 0.0f;
    float energy_coulomb = 0.0f;
    float selected_energy = 0.0f;
    float rest2_unscaled = 0.0f;
    float rest2_effective = 0.0f;
};

struct SITS_SOFT_CORE_POLICY
{
    float* atom_ene_LJ;
    VECTOR* frc;
    VECTOR* frc_enhancing;
    float* atom_energy;
    float* atom_energy_enhancing;
    LTMatrix3* atom_virial;
    LTMatrix3* atom_virial_enhancing;
    float* atom_direct_cf_energy;
    float pwwp_factor;

    __device__ __forceinline__ float Pair_Factor(const int mark_sum) const
    {
        if (mark_sum == 0)
        {
            return 1.0f;
        }
        if (mark_sum == 1)
        {
            return pwwp_factor;
        }
        return 0.0f;
    }

    template <bool need_force, bool need_energy, bool need_virial,
              bool need_coulomb>
    __device__ __forceinline__ void Accumulate_Pair(
        const int atom_j, const int local_atom_numbers, const float ij_factor,
        const VECTOR dr, const float frc_abs, const float pair_lj,
        const float pair_coulomb, const int mark_sum,
        SELECTIVE_PAIR_ACCUMULATOR* acc) const
    {
        const float factor = Pair_Factor(mark_sum);
        if (need_force)
        {
            VECTOR frc_lin = frc_abs * dr;
            acc->force = acc->force + frc_lin;
            acc->selected_force = acc->selected_force + factor * frc_lin;
            if (atom_j < local_atom_numbers)
            {
                atomicAdd(frc + atom_j, -frc_lin);
                atomicAdd(frc_enhancing + atom_j, -factor * frc_lin);
            }
            if (need_virial)
            {
                LTMatrix3 virial0 = Get_Virial_From_Force_Dis(frc_lin, dr);
                acc->virial = acc->virial + ij_factor * virial0;
                acc->selected_virial =
                    acc->selected_virial + ij_factor * factor * virial0;
            }
        }
        if (need_energy)
        {
            acc->energy_lj += ij_factor * pair_lj;
            if (need_coulomb)
            {
                acc->energy_coulomb += ij_factor * pair_coulomb;
            }
            acc->selected_energy +=
                ij_factor * factor * (pair_lj + pair_coulomb);
        }
    }

    template <bool need_force, bool need_energy, bool need_virial,
              bool need_coulomb>
    __device__ __forceinline__ void Write_Atom(
        const int atom_i, SELECTIVE_PAIR_ACCUMULATOR* acc) const
    {
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, acc->force, warpSize);
            Warp_Sum_To(frc_enhancing + atom_i, acc->selected_force, warpSize);
        }
        if (need_coulomb && need_energy)
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, acc->energy_coulomb,
                        warpSize);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, acc->energy_lj, warpSize);
#ifdef USE_GPU
            if (threadIdx.x == 0)
#endif
                atomicAdd(atom_ene_LJ + atom_i, acc->energy_lj);
            Warp_Sum_To(atom_energy_enhancing + atom_i, acc->selected_energy,
                        warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, acc->virial, warpSize);
            Warp_Sum_To(atom_virial_enhancing + atom_i, acc->selected_virial,
                        warpSize);
        }
    }
};

struct SITS_NORMAL_POLICY
{
    float* atom_ene_LJ;
    VECTOR* frc;
    VECTOR* frc_enhancing;
    float* atom_energy;
    float* atom_energy_enhancing;
    LTMatrix3* atom_virial;
    LTMatrix3* atom_virial_enhancing;
    float* atom_direct_cf_energy;
    float pwwp_factor;

    __device__ __forceinline__ float Pair_Factor(const int mark_sum) const
    {
        if (mark_sum == 0)
        {
            return 1.0f;
        }
        if (mark_sum == 1)
        {
            return pwwp_factor;
        }
        return 0.0f;
    }

    template <bool need_force, bool need_energy, bool need_virial,
              bool need_coulomb>
    __device__ __forceinline__ void Accumulate_Pair(
        const int atom_j, const int local_atom_numbers, const float ij_factor,
        const VECTOR dr, const float frc_abs, const float pair_lj,
        const float pair_coulomb, const int mark_sum,
        SELECTIVE_PAIR_ACCUMULATOR* acc) const
    {
        const float factor = Pair_Factor(mark_sum);
        if (need_force)
        {
            VECTOR frc_lin = frc_abs * dr;
            acc->force = acc->force + frc_lin;
            if (atom_j < local_atom_numbers)
            {
                atomicAdd(frc + atom_j, -frc_lin);
            }
            frc_lin = factor * frc_lin;
            acc->selected_force = acc->selected_force + frc_lin;
            if (need_virial)
            {
                LTMatrix3 virial0 = Get_Virial_From_Force_Dis(frc_lin, dr);
                acc->virial = acc->virial + ij_factor * virial0;
                acc->selected_virial =
                    acc->selected_virial + ij_factor * factor * virial0;
            }
        }
        if (need_coulomb && need_energy)
        {
            acc->energy_coulomb += ij_factor * pair_coulomb;
            acc->selected_energy += ij_factor * factor * pair_coulomb;
        }
        if (need_energy)
        {
            acc->energy_lj += ij_factor * pair_lj;
            acc->selected_energy += ij_factor * factor * pair_lj;
        }
    }

    template <bool need_force, bool need_energy, bool need_virial,
              bool need_coulomb>
    __device__ __forceinline__ void Write_Atom(
        const int atom_i, SELECTIVE_PAIR_ACCUMULATOR* acc) const
    {
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, acc->force, warpSize);
            Warp_Sum_To(frc_enhancing + atom_i, acc->selected_force, warpSize);
        }
        if (need_coulomb && need_energy)
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, acc->energy_coulomb,
                        warpSize);
        }
        if (need_energy)
        {
            Warp_Sum_To(atom_energy + atom_i, acc->energy_lj, warpSize);
#ifdef USE_GPU
            if (threadIdx.x == 0)
#endif
                atomicAdd(atom_ene_LJ + atom_i, acc->energy_lj);
            Warp_Sum_To(atom_energy_enhancing + atom_i, acc->selected_energy,
                        warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, acc->virial, warpSize);
            Warp_Sum_To(atom_virial_enhancing + atom_i, acc->selected_virial,
                        warpSize);
        }
    }
};

struct REST2_NORMAL_POLICY
{
    VECTOR* frc;
    float* atom_energy;
    LTMatrix3* atom_virial;
    float* atom_direct_cf_energy;
    float* atom_LJ_ene;
    float* rest2_unscaled_atom_energy;
    float* rest2_effective_atom_energy;
    float lambda_m;
    float sqrt_lambda_m;

    __device__ __forceinline__ float Pair_Scale(const int mark_sum) const
    {
        if (mark_sum == 0)
        {
            return lambda_m;
        }
        if (mark_sum == 1)
        {
            return sqrt_lambda_m;
        }
        return 1.0f;
    }

    template <bool need_force, bool need_energy, bool need_virial,
              bool need_coulomb>
    __device__ __forceinline__ void Accumulate_Pair(
        const int atom_j, const int local_atom_numbers, const float ij_factor,
        const VECTOR dr, const float frc_abs, const float pair_lj,
        const float pair_coulomb, const int mark_sum,
        SELECTIVE_PAIR_ACCUMULATOR* acc) const
    {
        const float scale = Pair_Scale(mark_sum);
        if (need_force)
        {
            VECTOR frc_lin = scale * frc_abs * dr;
            acc->force = acc->force + frc_lin;
            if (atom_j < local_atom_numbers)
            {
                atomicAdd(frc + atom_j, -frc_lin);
            }
            if (need_virial)
            {
                acc->virial =
                    acc->virial -
                    ij_factor * Get_Virial_From_Force_Dis(frc_lin, dr);
            }
        }
        if (need_energy)
        {
            const float pair_total = pair_lj + pair_coulomb;
            acc->energy_lj += ij_factor * scale * pair_lj;
            if (need_coulomb)
            {
                acc->energy_coulomb += ij_factor * scale * pair_coulomb;
            }
            if (mark_sum < 2)
            {
                acc->rest2_unscaled += ij_factor * pair_total;
                acc->rest2_effective += ij_factor * scale * pair_total;
            }
        }
    }

    template <bool need_force, bool need_energy, bool need_virial,
              bool need_coulomb>
    __device__ __forceinline__ void Write_Atom(
        const int atom_i, SELECTIVE_PAIR_ACCUMULATOR* acc) const
    {
        if (need_force)
        {
            Warp_Sum_To(frc + atom_i, acc->force, warpSize);
        }
        if (need_energy)
        {
            float energy_total = acc->energy_lj;
            if (need_coulomb)
            {
                energy_total += acc->energy_coulomb;
            }
            Warp_Sum_To(atom_energy + atom_i, energy_total, warpSize);
            Warp_Sum_To(atom_LJ_ene + atom_i, acc->energy_lj, warpSize);
            Warp_Sum_To(rest2_unscaled_atom_energy + atom_i,
                        acc->rest2_unscaled, warpSize);
            Warp_Sum_To(rest2_effective_atom_energy + atom_i,
                        acc->rest2_effective, warpSize);
        }
        if (need_coulomb && need_energy)
        {
            Warp_Sum_To(atom_direct_cf_energy + atom_i, acc->energy_coulomb,
                        warpSize);
        }
        if (need_virial)
        {
            Warp_Sum_To(atom_virial + atom_i, acc->virial, warpSize);
        }
    }
};

using REST2_SOFT_CORE_POLICY = REST2_NORMAL_POLICY;
