/*
 * This file is part of the Lipid Swap MC extension of GROMACS.
 *
 * Copyright 2026 Wooseok Jung <wshr0307@sogang.ac.kr>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * This is not an official GROMACS file; see the COPYING file in the
 * top-level directory for the license of GROMACS itself.
 */
/*
 * Lipid Swap Monte Carlo for the legacy simulator.
 *
 * Performs DPPC <-> DIPC residue-swap MC moves between inner and outer
 * leaflets of a MARTINI vesicle. Designed to be invoked from the legacy
 * do_md() main loop so that the standard MD pipeline (vsite construction,
 * constraints, integrator, trajectory output) is preserved exactly.
 *
 * Supported setup (checked at start-up, mdrun stops otherwise):
 * a single rank (mdrun -ntmpi 1), update on the CPU (mdrun -update cpu),
 * 12-bead MARTINI 2 DPPC and DIPC,
 * LJ cut-off with potential-shift or no modifier, and lipidswap-mc-nst a
 * multiple of the nstlist used at run time.
 */
#ifndef GMX_MDRUN_LIPIDSWAP_MC_H
#define GMX_MDRUN_LIPIDSWAP_MC_H

#include <array>
#include <cstdio>
#include <random>
#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/real.h"

struct gmx_enerdata_t;
struct gmx_enfrot;
struct gmx_localtop_t;
struct gmx_mtop_t;
struct gmx_wallcycle;
struct pull_t;
struct t_commrec;
struct t_forcerec;
struct t_inputrec;
struct t_nrnb;
struct t_state;

namespace gmx
{
class Awh;
class ForceBuffers;
class ImdSession;
class MDAtoms;
struct MDModulesNotifiers;
class MdrunScheduleWorkload;
class VirtualSitesHandler;

/*! \internal
 * Lipid Swap Monte Carlo (legacy do_md integration).
 *
 * Owns its own residue/leaflet bookkeeping and swap.log handle.
 * apply() does one MC block (mc_nsteps_ swap attempts) at the requested step.
 */
class LipidSwapMC
{
public:
    LipidSwapMC(FILE*                     fplog,
                const t_commrec*          cr,
                const t_inputrec*         inputrec,
                const MDModulesNotifiers& mdModulesNotifiers,
                const MDAtoms*            mdAtoms,
                t_nrnb*                   nrnb,
                t_forcerec*               fr,
                gmx_wallcycle*            wcycle,
                MdrunScheduleWorkload*    runScheduleWork,
                VirtualSitesHandler*      vsite,
                ImdSession*               imdSession,
                pull_t*                   pull_work,
                gmx_enfrot*               enforcedRotation,
                const gmx_mtop_t&         globalTopology,
                const gmx_localtop_t*     localTopology,
                t_state*                  state,
                gmx_enerdata_t*           enerd,
                ForceBuffers*             forceBuffers);

    ~LipidSwapMC();

    /*! Should an MC block run at this step? */
    bool shouldRun(int64_t step) const
    {
        return mc_freq_ > 0 && step > 0 && (step % mc_freq_ == 0);
    }

    /*! Run one MC block (mc_nsteps_ swap attempts) at the current state. */
    void apply(int64_t step, double time);

    /*! Final summary printed to fplog at the end of the run. */
    void writeSummary();

private:
    struct ResidueInfo
    {
        std::vector<int> atomIndices;
        int              po4Idx = -1;
        int              c4aIdx = -1;
        int              c4bIdx = -1;
    };

    // Simple orthorhombic cell list for environment LJ scan.
    struct CellList
    {
        real Lx = 0, Ly = 0, Lz = 0;
        real invLx = 0, invLy = 0, invLz = 0;
        int  nx = 0, ny = 0, nz = 0;
        std::vector<std::vector<int>> cells;
        std::vector<int>              atomCell;
        bool valid = false;

        int cellIndex(int ix, int iy, int iz) const
        {
            ix = (ix % nx + nx) % nx;
            iy = (iy % ny + ny) % ny;
            iz = (iz % nz + nz) % nz;
            return ix + iy * nx + iz * nx * ny;
        }
        int cellOf(const RVec& r) const
        {
            real fx = r[XX] * invLx;
            fx -= std::floor(fx);
            real fy = r[YY] * invLy;
            fy -= std::floor(fy);
            real fz = r[ZZ] * invLz;
            fz -= std::floor(fz);
            int ix = std::min(nx - 1, std::max(0, int(fx * nx)));
            int iy = std::min(ny - 1, std::max(0, int(fy * ny)));
            int iz = std::min(nz - 1, std::max(0, int(fz * nz)));
            return ix + iy * nx + iz * nx * ny;
        }
    };

    void buildResidueInfo(const gmx_mtop_t& top);
    void initLipidParams(const gmx_mtop_t& top);
    void classifyLeaflets();
    void buildCellList();
    void updateCellListAtom(int globalAtomIdx);
    real computeEnergy(int64_t step, double time);
    real computeSwapDeltaE(const ResidueInfo& dppc, const ResidueInfo& dipc) const;
    void openLogFile();
    /*! Per-block summary statistics (one struct populated and written per MC block). */
    struct BlockStats
    {
        int     att         = 0;       // attempts in block
        int     acc         = 0;       // accepted in block
        int     r0          = 0;       // rule-0 accepted in block
        int     r1          = 0;       // rule-1 accepted in block
        double  dE_sum      = 0;       // sum of all attempted ΔE
        double  dE_min      = 0;       // most negative ΔE
        double  dE_max      = 0;       // most positive ΔE
        double  dE_acc_sum  = 0;       // sum of ΔE for accepted attempts
        double  mh_sum      = 0;       // sum of MH ratios (sanity, ~1.0)
        bool    haveAny     = false;
    };
    void writeBlockLog(int64_t step, double time, int blockIdx, const BlockStats& bs);

    // Optional per-accepted-swap coordinate dump for visualization (OVITO).
    // Enabled only when the env var GMX_LIPIDSWAP_TRAJ is set; default is off,
    // so production runs are byte-for-byte unaffected.  Writes a LAMMPS custom
    // dump (mc_swaps.lammpstrj) with columns: id type x y z swap, where swap=1
    // marks the 24 atoms of the swap shown in that frame.
    void buildVizAtoms(const gmx_mtop_t& top);
    void openSwapTraj();
    void writeSwapFrame(const ResidueInfo& dppc, const ResidueInfo& dipc);

    // ---- legacy state pointers ----
    FILE*                     fplog_;
    const t_commrec*          cr_;
    const t_inputrec*         inputrec_;
    const MDModulesNotifiers& mdModulesNotifiers_;
    const MDAtoms*            mdAtoms_;
    t_nrnb*                   nrnb_;
    t_forcerec*               fr_;
    gmx_wallcycle*            wcycle_;
    MdrunScheduleWorkload*    runScheduleWork_;
    VirtualSitesHandler*      vsite_;
    ImdSession*               imdSession_;
    pull_t*                   pull_work_;
    gmx_enfrot*               enforcedRotation_;
    const gmx_localtop_t*     localTopology_;
    t_state*                  state_;
    gmx_enerdata_t*           enerd_;
    ForceBuffers*             forceBuffers_;

    // ---- MC parameters ----
    const int    mc_freq_;
    const int    mc_nsteps_;
    const double beta_;

    std::vector<ResidueInfo> dppc_residues_;
    std::vector<ResidueInfo> dipc_residues_;

    // Per-bead LJ atom-type indices for one DPPC and one DIPC molecule.
    // Layout assumed: 12 beads (NC3 PO4 GL1 GL2 C1A C2A/D2A C3A/D3A C4A C1B C2B/D2B C3B/D3B C4B).
    std::array<int, 12> dppcBeadAtype_{};
    std::array<int, 12> dipcBeadAtype_{};
    int                 ntype_            = 0;
    real                rvdw_             = 0;
    real                rvdw2_            = 0;
    bool                potShiftLJ_       = false;
    // Validate local ΔE against full do_force for the first 5 attempts of the
    // run that fall on an energy-calculation step.  Set from
    // GMX_LIPIDSWAP_VALIDATE; adds 2 force calls per attempt.
    bool validateDeltaE_ = false;

    mutable CellList cellList_;

    std::vector<int> inner_dppc_;
    std::vector<int> outer_dppc_;
    std::vector<int> inner_dipc_;
    std::vector<int> outer_dipc_;

    int64_t total_          = 0;
    int64_t accepted_       = 0;
    int64_t rule0_accepted_ = 0;
    int64_t rule1_accepted_ = 0;

    unsigned int                                   seedUsed_;  // seed actually used (logged)
    mutable std::mt19937                           rng_;
    mutable std::uniform_real_distribution<double> uniform01_{ 0.0, 1.0 };

    FILE* mcLogFile_     = nullptr;
    bool  headerWritten_ = false;

    // ---- optional swap-trajectory output (visualization only) ----
    bool             swapTrajEnabled_ = false;   // set from GMX_LIPIDSWAP_TRAJ
    FILE*            swapTrajFile_    = nullptr;  // mc_swaps.lammpstrj handle
    int64_t          swapFrameIdx_    = 0;        // running frame counter
    std::vector<int> vizAtoms_;                   // global indices of lipid atoms to dump (sorted)
    std::vector<int> vizType_;                    // parallel type code: 1=DPPC, 2=DIPC, 3=CHOL
};

} // namespace gmx

#endif // GMX_MDRUN_LIPIDSWAP_MC_H
