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
 * See lipidswap_mc.h for the design rationale.
 */
#include "gmxpre.h"

#include "lipidswap_mc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <iterator>
#include <utility>

#include "gromacs/domdec/dlbtiming.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forcebuffers.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"

struct gmx_edsam;
struct gmx_multisim_t;

namespace gmx
{

namespace
{

// 12-bead MARTINI 2 layout, 0-indexed:
// NC3 PO4 GL1 GL2 C1A C2A/D2A C3A/D3A C4A C1B C2B/D2B C3B/D3B C4B
constexpr int c_numBeads = 12;

// Positions where the DPPC and DIPC bead types differ:
// 2nd and 3rd bead of each tail, C1 -> C4.
constexpr int c_diffPos[4] = { 5, 6, 9, 10 };

// Intra-residue bonded list (0-indexed).  Same for DPPC and DIPC.
constexpr std::pair<int, int> c_bonded[11] = { { 0, 1 }, { 1, 2 }, { 2, 3 },  { 2, 4 },
                                               { 4, 5 }, { 5, 6 }, { 6, 7 },  { 3, 8 },
                                               { 8, 9 }, { 9, 10 }, { 10, 11 } };

// The four G96 angles (function 2) whose parameters differ: atom indices
// (i,j,k) within the residue and (theta0 in degrees, kc) for DPPC and DIPC.
// MARTINI 2.0/2.2 values; initLipidParams() checks them against the topology.
struct DiffAngle
{
    int    i, j, k;
    double theta0Dppc;
    real   kcDppc;
    double theta0Dipc;
    real   kcDipc;
};
constexpr DiffAngle c_diffAngles[4] = {
    { 4, 5, 6, 180.0, 25.0_real, 100.0, 10.0_real },  //  5- 6- 7
    { 5, 6, 7, 180.0, 25.0_real, 120.0, 45.0_real },  //  6- 7- 8
    { 8, 9, 10, 180.0, 25.0_real, 100.0, 10.0_real }, //  9-10-11
    { 9, 10, 11, 180.0, 25.0_real, 120.0, 45.0_real } // 10-11-12
};

bool isDiffPos(int k)
{
    return std::find(std::begin(c_diffPos), std::end(c_diffPos), k) != std::end(c_diffPos);
}

bool isBondedPair(int a, int b)
{
    if (a > b)
    {
        std::swap(a, b);
    }
    for (const auto& p : c_bonded)
    {
        if (p.first == a && p.second == b)
        {
            return true;
        }
    }
    return false;
}

//! Whether two parameter sets of interaction type \p ftype are equal.
//! grompp gives each molecule type its own parameter entries, so equal
//! parameters in DPPC and DIPC usually have different type indices.
bool sameParams(int ftype, const t_iparams& a, const t_iparams& b)
{
    switch (ftype)
    {
        case F_PDIHS:
        case F_PIDIHS:
        case F_ANGRES:
        case F_ANGRESZ:
            return a.pdihs.phiA == b.pdihs.phiA && a.pdihs.cpA == b.pdihs.cpA
                   && a.pdihs.mult == b.pdihs.mult && a.pdihs.phiB == b.pdihs.phiB
                   && a.pdihs.cpB == b.pdihs.cpB;
        default:
            // All other listed interaction types store NRFP(ftype) reals.
            return std::memcmp(&a, &b, NRFP(ftype) * sizeof(real)) == 0;
    }
}

//! MC temperature: the common ref-t of all temperature-coupling groups.
double lipidSwapTemperature(const t_inputrec& ir)
{
    if (ir.etc == TemperatureCoupling::No && !EI_SD(ir.eI))
    {
        gmx_fatal(FARGS,
                  "Lipid swap MC needs a temperature: use temperature coupling (tcoupl) "
                  "or integrator sd.");
    }
    if (ir.opts.ngtc < 1)
    {
        gmx_fatal(FARGS, "Lipid swap MC needs at least one temperature-coupling group.");
    }
    const real t0 = ir.opts.ref_t[0];
    for (int g = 1; g < ir.opts.ngtc; g++)
    {
        if (std::abs(ir.opts.ref_t[g] - t0) > 1e-6 * t0)
        {
            gmx_fatal(FARGS,
                      "Lipid swap MC needs the same ref-t for all tc-grps "
                      "(group 0: %g K, group %d: %g K).",
                      t0,
                      g,
                      ir.opts.ref_t[g]);
        }
    }
    if (t0 <= 0)
    {
        gmx_fatal(FARGS, "Lipid swap MC needs ref-t > 0 (got %g K).", t0);
    }
    return t0;
}

//! Seed for the MC random number generator; a negative seed draws one.
unsigned int resolveSeed(int seed)
{
    return seed >= 0 ? static_cast<unsigned int>(seed) : std::random_device{}();
}

} // namespace

LipidSwapMC::LipidSwapMC(FILE*                     fplog,
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
                         ForceBuffers*             forceBuffers) :
    fplog_(fplog),
    cr_(cr),
    inputrec_(inputrec),
    mdModulesNotifiers_(mdModulesNotifiers),
    mdAtoms_(mdAtoms),
    nrnb_(nrnb),
    fr_(fr),
    wcycle_(wcycle),
    runScheduleWork_(runScheduleWork),
    vsite_(vsite),
    imdSession_(imdSession),
    pull_work_(pull_work),
    enforcedRotation_(enforcedRotation),
    localTopology_(localTopology),
    state_(state),
    enerd_(enerd),
    forceBuffers_(forceBuffers),
    mc_freq_(inputrec->nst_lipidswap),
    mc_nsteps_(inputrec->nstep_lipidswap),
    beta_(1.0 / (gmx::c_boltz * lipidSwapTemperature(*inputrec))),
    seedUsed_(resolveSeed(inputrec->ld_seed_lipidswap)),
    rng_(seedUsed_)
{
    // The residue bookkeeping below uses global atom indices, which equal the
    // local ones only when a single rank holds the whole system.
    if (PAR(cr))
    {
        gmx_fatal(FARGS,
                  "Lipid swap MC supports only a single rank. Run mdrun with -ntmpi 1 "
                  "(thread-MPI) or with a single MPI process.");
    }
    // MC moves the coordinates on the CPU; with GPU update the MD loop keeps
    // using the device copy and never sees the swap.
    if (runScheduleWork->simulationWork.useGpuUpdate)
    {
        gmx_fatal(FARGS,
                  "Lipid swap MC does not support update on the GPU. Run mdrun with "
                  "-update cpu.");
    }
    if (mc_freq_ <= 0 || mc_nsteps_ <= 0)
    {
        gmx_fatal(FARGS,
                  "lipidswap-mc-nst (%d) and lipidswap-mc-nstep (%d) must be positive.",
                  mc_freq_,
                  mc_nsteps_);
    }
    // do_force() rebuilds the pair list only on neighbor-search steps.  An MC
    // block that moves lipids on any other step leaves the moved beads paired
    // with their old neighbours until the next search step.
    if (mc_freq_ % inputrec->nstlist != 0)
    {
        gmx_fatal(FARGS,
                  "lipidswap-mc-nst (%d) is not a multiple of nstlist (%d) used in this run. "
                  "MC blocks must fall on neighbor-search steps. mdrun may increase nstlist "
                  "(to 20, 25, 40, 50, 80 or 100, on CPUs and GPUs); use a multiple of 400 "
                  "for lipidswap-mc-nst, or fix nstlist with mdrun -nstlist.",
                  mc_freq_,
                  inputrec->nstlist);
    }
    if (inputrec->efep != FreeEnergyPerturbationType::No)
    {
        gmx_fatal(FARGS, "Lipid swap MC does not support free-energy perturbation.");
    }

    buildResidueInfo(globalTopology);
    initLipidParams(globalTopology);
    if (const char* env = std::getenv("GMX_LIPIDSWAP_VALIDATE"))
    {
        validateDeltaE_ = (env[0] == '1' || env[0] == 'y' || env[0] == 'Y');
    }
    if (validateDeltaE_ && fplog_ && inputrec->nstcalcenergy > 0
        && mc_freq_ % inputrec->nstcalcenergy != 0)
    {
        fprintf(fplog_,
                "Note: GMX_LIPIDSWAP_VALIDATE compares against the total energy, which is "
                "computed only on energy steps (nstcalcenergy = %d). MC blocks on other steps "
                "are not validated.\n",
                inputrec->nstcalcenergy);
    }
    // Visualization-only: lipid/cholesterol coordinates to mc_swaps.lammpstrj
    // after each accepted swap.  Off unless GMX_LIPIDSWAP_TRAJ is set.
    if (const char* env = std::getenv("GMX_LIPIDSWAP_TRAJ"))
    {
        swapTrajEnabled_ = (env[0] == '1' || env[0] == 'y' || env[0] == 'Y');
    }
    // Classify leaflets once, here; accepted swaps relabel membership directly.
    // classifyLeaflets() ignores PBC, so the starting structure must be whole
    // and centred in the box.
    classifyLeaflets();
    openLogFile();
    if (swapTrajEnabled_)
    {
        buildVizAtoms(globalTopology);
        openSwapTraj();
    }

    if (fplog_)
    {
        fprintf(fplog_,
                "\nLipid Swap MC: found %zu DPPC and %zu DIPC residues.\n"
                "  MC frequency: every %d MD steps, %d moves per block.\n"
                "  Temperature : %.2f K  (beta = %.4g mol/kJ)\n"
                "  RNG seed    : %u%s\n"
                "  Swap traj   : %s\n\n",
                dppc_residues_.size(),
                dipc_residues_.size(),
                mc_freq_,
                mc_nsteps_,
                1.0 / (gmx::c_boltz * beta_),
                beta_,
                seedUsed_,
                inputrec->ld_seed_lipidswap < 0 ? " (drawn from random_device)" : "",
                swapTrajEnabled_ ? "mc_swaps.lammpstrj (per accepted swap)" : "off");
    }
}

LipidSwapMC::~LipidSwapMC()
{
    if (mcLogFile_)
    {
        std::fclose(mcLogFile_);
    }
    if (swapTrajFile_)
    {
        std::fclose(swapTrajFile_);
        swapTrajFile_ = nullptr;
    }
}

void LipidSwapMC::writeSummary()
{
    if (fplog_ && total_ > 0)
    {
        fprintf(fplog_,
                "\nLipid Swap MC summary: %lld accepted / %lld total (ratio %.4f)\n"
                "  Rule-0 (DPPC-in <-> DIPC-out): %lld accepted\n"
                "  Rule-1 (DIPC-in <-> DPPC-out): %lld accepted\n\n",
                (long long)accepted_,
                (long long)total_,
                static_cast<double>(accepted_) / total_,
                (long long)rule0_accepted_,
                (long long)rule1_accepted_);
    }
}

void LipidSwapMC::buildResidueInfo(const gmx_mtop_t& top)
{
    int globalAtomOffset = 0;

    for (const gmx_molblock_t& molblock : top.molblock)
    {
        const gmx_moltype_t& moltype  = top.moltype[molblock.type];
        const t_atoms&       atoms    = moltype.atoms;
        const int            nAtomsMol = atoms.nr;

        for (int mol = 0; mol < molblock.nmol; mol++)
        {
            const int molOffset = globalAtomOffset + mol * nAtomsMol;

            int         curResInd = -1;
            bool        isDPPC    = false;
            bool        isDIPC    = false;
            ResidueInfo info;

            auto flushResidue = [&]()
            {
                if (curResInd < 0) return;
                if (isDPPC) dppc_residues_.push_back(std::move(info));
                else if (isDIPC) dipc_residues_.push_back(std::move(info));
                info = ResidueInfo{};
            };

            for (int a = 0; a < nAtomsMol; a++)
            {
                int newResInd = atoms.atom[a].resind;
                if (newResInd != curResInd)
                {
                    flushResidue();
                    curResInd        = newResInd;
                    const char* rname = *(atoms.resinfo[newResInd].name);
                    isDPPC           = (std::strcmp(rname, "DPPC") == 0);
                    isDIPC           = (std::strcmp(rname, "DIPC") == 0);
                }

                if (isDPPC || isDIPC)
                {
                    const int globalIdx = molOffset + a;
                    info.atomIndices.push_back(globalIdx);

                    const char* aname = *(atoms.atomname[a]);
                    if (std::strcmp(aname, "PO4") == 0) info.po4Idx = globalIdx;
                    if (std::strcmp(aname, "C4A") == 0) info.c4aIdx = globalIdx;
                    if (std::strcmp(aname, "C4B") == 0) info.c4bIdx = globalIdx;
                }
            }
            flushResidue();
        }

        globalAtomOffset += molblock.nmol * nAtomsMol;
    }
}

void LipidSwapMC::initLipidParams(const gmx_mtop_t& top)
{
    // Locate the DPPC / DIPC moltypes, extract the LJ atom-type index of each
    // of the 12 beads and the cutoff/modifier, and check that the topology
    // matches every assumption made by computeSwapDeltaE().  Any mismatch
    // stops mdrun: there is no fallback energy path.
    auto findMoltype = [&](const char* resname) -> int
    {
        for (size_t i = 0; i < top.moltype.size(); i++)
        {
            const t_atoms& atoms = top.moltype[i].atoms;
            if (atoms.nr == 0 || atoms.resinfo == nullptr) continue;
            const char* rname = *(atoms.resinfo[0].name);
            if (std::strcmp(rname, resname) == 0) return static_cast<int>(i);
        }
        return -1;
    };

    const int idxDPPC = findMoltype("DPPC");
    const int idxDIPC = findMoltype("DIPC");
    if (idxDPPC < 0 || idxDIPC < 0)
    {
        gmx_fatal(FARGS, "Lipid swap MC needs both a DPPC and a DIPC molecule type.");
    }

    const gmx_moltype_t& mDPPC = top.moltype[idxDPPC];
    const gmx_moltype_t& mDIPC = top.moltype[idxDIPC];
    const t_atoms&       aDPPC = mDPPC.atoms;
    const t_atoms&       aDIPC = mDIPC.atoms;
    if (aDPPC.nr != c_numBeads || aDIPC.nr != c_numBeads)
    {
        gmx_fatal(FARGS,
                  "Lipid swap MC needs 12-bead MARTINI 2 DPPC and DIPC (got %d / %d beads).",
                  aDPPC.nr,
                  aDIPC.nr);
    }

    for (int k = 0; k < c_numBeads; k++)
    {
        dppcBeadAtype_[k] = aDPPC.atom[k].type;
        dipcBeadAtype_[k] = aDIPC.atom[k].type;
        // Only the four tail positions may change LJ type; the local energy
        // difference ignores every other position.
        if (!isDiffPos(k) && dppcBeadAtype_[k] != dipcBeadAtype_[k])
        {
            gmx_fatal(FARGS,
                      "Lipid swap MC: DPPC and DIPC bead %d (%s / %s) have different atom types; "
                      "only beads 6, 7, 10 and 11 may differ.",
                      k + 1,
                      *(aDPPC.atomname[k]),
                      *(aDIPC.atomname[k]));
        }
        // Coulomb terms cancel only if every bead keeps its charge.
        if (aDPPC.atom[k].q != aDIPC.atom[k].q)
        {
            gmx_fatal(FARGS,
                      "Lipid swap MC: DPPC and DIPC bead %d (%s / %s) have different charges.",
                      k + 1,
                      *(aDPPC.atomname[k]),
                      *(aDIPC.atomname[k]));
        }
    }

    // Bonded interactions: identical atoms in both molecules, identical
    // parameter values except for the four G96 angles in c_diffAngles.
    const auto& iparams = top.ffparams.iparams;
    for (int ftype = 0; ftype < F_NRE; ftype++)
    {
        const auto& ilD  = mDPPC.ilist[ftype].iatoms;
        const auto& ilI  = mDIPC.ilist[ftype].iatoms;
        const int   nral = NRAL(ftype);
        if (ilD.size() != ilI.size())
        {
            gmx_fatal(FARGS,
                      "Lipid swap MC: DPPC and DIPC have a different number of %s interactions.",
                      interaction_function[ftype].longname);
        }
        for (size_t e = 0; e < ilD.size(); e += 1 + nral)
        {
            for (int a = 1; a <= nral; a++)
            {
                if (ilD[e + a] != ilI[e + a])
                {
                    gmx_fatal(FARGS,
                              "Lipid swap MC: DPPC and DIPC differ in the atoms of a %s "
                              "interaction.",
                              interaction_function[ftype].longname);
                }
            }
            if (sameParams(ftype, iparams[ilD[e]], iparams[ilI[e]]))
            {
                continue;
            }
            const bool isDiffAngle =
                    ftype == F_G96ANGLES
                    && std::any_of(std::begin(c_diffAngles),
                                   std::end(c_diffAngles),
                                   [&](const DiffAngle& A)
                                   { return ilD[e + 1] == A.i && ilD[e + 2] == A.j && ilD[e + 3] == A.k; });
            if (!isDiffAngle)
            {
                gmx_fatal(FARGS,
                          "Lipid swap MC: DPPC and DIPC have different parameters for a %s "
                          "interaction between beads %d-%d; only the four tail G96 angles may "
                          "differ.",
                          interaction_function[ftype].longname,
                          ilD[e + 1] + 1,
                          ilD[e + 2] + 1);
            }
        }
    }
    auto checkAngle = [&](const gmx_moltype_t& m, const DiffAngle& A, double theta0, real kc, const char* name)
    {
        const auto& il = m.ilist[F_G96ANGLES].iatoms;
        for (size_t e = 0; e < il.size(); e += 4)
        {
            if (il[e + 1] == A.i && il[e + 2] == A.j && il[e + 3] == A.k)
            {
                // grompp stores G96 angles as (cos theta0, kc).
                const auto&  h    = iparams[il[e]].harmonic;
                const double cos0 = std::cos(theta0 * M_PI / 180.0);
                if (std::abs(h.rA - cos0) > 1e-5 || std::abs(h.krA - kc) > 1e-4)
                {
                    gmx_fatal(FARGS,
                              "Lipid swap MC: %s G96 angle %d-%d-%d is (%g deg, %g) in the "
                              "topology but the code assumes (%g deg, %g) (MARTINI 2).",
                              name,
                              A.i + 1,
                              A.j + 1,
                              A.k + 1,
                              std::acos(std::clamp(static_cast<double>(h.rA), -1.0, 1.0)) * 180.0 / M_PI,
                              h.krA,
                              theta0,
                              kc);
                }
                return;
            }
        }
        gmx_fatal(FARGS,
                  "Lipid swap MC: %s has no G96 angle %d-%d-%d.", name, A.i + 1, A.j + 1, A.k + 1);
    };
    for (const auto& A : c_diffAngles)
    {
        checkAngle(mDPPC, A, A.theta0Dppc, A.kcDppc, "DPPC");
        checkAngle(mDIPC, A, A.theta0Dipc, A.kcDipc, "DIPC");
    }

    // Intra-residue LJ in computeSwapDeltaE() skips exactly the bonded pairs
    // (nrexcl = 1).  Check that the topology excludes the same pairs.
    auto checkExclusions = [&](const gmx_moltype_t& m, const char* name)
    {
        for (int a = 0; a < c_numBeads; a++)
        {
            for (int b = 0; b < c_numBeads; b++)
            {
                if (a == b) continue;
                const auto excl     = m.excls[a];
                const bool excluded = std::find(excl.begin(), excl.end(), b) != excl.end();
                if (excluded != isBondedPair(a, b))
                {
                    gmx_fatal(FARGS,
                              "Lipid swap MC: the exclusions of %s beads %d-%d do not match "
                              "nrexcl = 1 over the MARTINI 2 bond list.",
                              name,
                              a + 1,
                              b + 1);
                }
            }
        }
    };
    checkExclusions(mDPPC, "DPPC");
    checkExclusions(mDIPC, "DIPC");

    // The local LJ energy is a plain cut-off with an optional potential shift.
    const auto& ic = *fr_->ic;
    if (ic.vdwtype != VanDerWaalsType::Cut
        || (ic.vdw_modifier != InteractionModifiers::PotShift
            && ic.vdw_modifier != InteractionModifiers::None))
    {
        gmx_fatal(FARGS,
                  "Lipid swap MC supports only vdw-type = cut-off with vdw-modifier = "
                  "potential-shift or none (got %s / %s).",
                  enumValueToString(ic.vdwtype),
                  enumValueToString(ic.vdw_modifier));
    }

    ntype_      = fr_->ntype;
    rvdw_       = ic.rvdw;
    rvdw2_      = rvdw_ * rvdw_;
    potShiftLJ_ = (ic.vdw_modifier == InteractionModifiers::PotShift);

    if (fplog_)
    {
        fprintf(fplog_,
                "LipidSwapMC local dE: ntype=%d, rvdw=%.3f nm, modifier=%s.\n"
                "  DPPC bead types:",
                ntype_, rvdw_,
                potShiftLJ_ ? "PotShift" : "None");
        for (int k = 0; k < c_numBeads; k++) fprintf(fplog_, " %d", dppcBeadAtype_[k]);
        fprintf(fplog_, "\n  DIPC bead types:");
        for (int k = 0; k < c_numBeads; k++) fprintf(fplog_, " %d", dipcBeadAtype_[k]);
        fprintf(fplog_, "\n");
    }
}

void LipidSwapMC::buildCellList()
{
    const auto&   xVec = state_->x;
    const matrix& box  = state_->box;
    const real    rvdw = rvdw_;

    cellList_.valid = false;

    // Only orthorhombic supported; otherwise fall back to brute-force in sweepEnv.
    const real eps = 1.0e-6_real;
    if (std::abs(box[XX][YY]) > eps || std::abs(box[XX][ZZ]) > eps
        || std::abs(box[YY][XX]) > eps || std::abs(box[YY][ZZ]) > eps
        || std::abs(box[ZZ][XX]) > eps || std::abs(box[ZZ][YY]) > eps)
    {
        return;
    }

    cellList_.Lx    = box[XX][XX];
    cellList_.Ly    = box[YY][YY];
    cellList_.Lz    = box[ZZ][ZZ];
    if (cellList_.Lx <= 0 || cellList_.Ly <= 0 || cellList_.Lz <= 0) return;
    cellList_.invLx = 1.0_real / cellList_.Lx;
    cellList_.invLy = 1.0_real / cellList_.Ly;
    cellList_.invLz = 1.0_real / cellList_.Lz;
    cellList_.nx    = std::max(3, int(cellList_.Lx / rvdw));
    cellList_.ny    = std::max(3, int(cellList_.Ly / rvdw));
    cellList_.nz    = std::max(3, int(cellList_.Lz / rvdw));

    const int   nCells  = cellList_.nx * cellList_.ny * cellList_.nz;
    const auto* mdatoms = mdAtoms_->mdatoms();
    const int   nAtoms  = static_cast<int>(mdatoms->typeA.size());

    cellList_.cells.assign(nCells, {});
    cellList_.atomCell.assign(nAtoms, -1);

    for (int i = 0; i < nAtoms; i++)
    {
        const int c = cellList_.cellOf(xVec[i]);
        cellList_.cells[c].push_back(i);
        cellList_.atomCell[i] = c;
    }

    cellList_.valid = true;
}

void LipidSwapMC::updateCellListAtom(int i)
{
    if (!cellList_.valid) return;
    const int oldC = cellList_.atomCell[i];
    const int newC = cellList_.cellOf(state_->x[i]);
    if (oldC == newC) return;
    auto& v  = cellList_.cells[oldC];
    auto  it = std::find(v.begin(), v.end(), i);
    if (it != v.end()) v.erase(it);
    cellList_.cells[newC].push_back(i);
    cellList_.atomCell[i] = newC;
}

/*! Local potential energy change of swapping one DPPC with one DIPC.
 *
 * The two lipids differ only in bead type and angle parameters at four
 * positions, so the swap is equivalent to interchanging those at fixed
 * coordinates.  Bond, Coulomb and improper terms then cancel exactly, leaving
 * the angle and LJ terms accumulated as (1)-(4) below.
 */
real LipidSwapMC::computeSwapDeltaE(const ResidueInfo& dppc, const ResidueInfo& dipc) const
{
    // Bead positions, bonds and differing angles: c_diffPos, c_bonded and
    // c_diffAngles at the top of this file, checked by initLipidParams().
    static const real kDeg2Rad = M_PI / 180.0;

    const auto&  xVec    = state_->x;
    const auto*  mdatoms = mdAtoms_->mdatoms();
    const int*   atype   = mdatoms->typeA.data();
    const real*  nbfp    = fr_->nbfp.data();
    const int    ntype   = ntype_;
    const real   rvdw2   = rvdw2_;
    const bool   potShift = potShiftLJ_;
    const int    nAtoms  = static_cast<int>(mdatoms->typeA.size());

    t_pbc pbc;
    set_pbc(&pbc, inputrec_->pbcType, state_->box);

    // LJ shift constants (only used if potShift)
    real shiftIr6 = 0, shiftIr12 = 0;
    if (potShift)
    {
        const real ir2c = 1.0_real / rvdw2;
        shiftIr6        = ir2c * ir2c * ir2c;
        shiftIr12       = shiftIr6 * shiftIr6;
    }

    // V_LJ (with optional pot-shift) given pre-scaled nbfp (c6_*6, c12_*12).
    auto ljEnergy = [&](int ai, int aj, real r2) -> real
    {
        const int  idx   = 2 * (ai * ntype + aj);
        const real c6    = nbfp[idx]     / 6.0_real;
        const real c12   = nbfp[idx + 1] / 12.0_real;
        const real ir2   = 1.0_real / r2;
        const real ir6   = ir2 * ir2 * ir2;
        const real ir12  = ir6 * ir6;
        real V = c12 * ir12 - c6 * ir6;
        if (potShift)
        {
            V -= c12 * shiftIr12 - c6 * shiftIr6;
        }
        return V;
    };

    // Atoms of both swap residues.  Excluded from the environment loop (1)
    // because their types change too; their pairs are handled by (3) and (4).
    std::array<int, 24> swapAtoms{};
    for (int b = 0; b < 12; b++)
    {
        swapAtoms[b]      = dppc.atomIndices[b];
        swapAtoms[b + 12] = dipc.atomIndices[b];
    }
    std::sort(swapAtoms.begin(), swapAtoms.end());
    auto inSwap = [&](int j)
    {
        return std::binary_search(swapAtoms.begin(), swapAtoms.end(), j);
    };

    real dE = 0;

    // -------- (1) Environment LJ delta at each differing-atom location --------
    // For each bead position p where DPPC-type != DIPC-type:
    //   - At DPPC's current position xVec[dppc.atom[p]]: type changes DPPC->DIPC
    //   - At DIPC's current position xVec[dipc.atom[p]]: type changes DIPC->DPPC
    //
    // Cell-list path (orthorhombic): only scan 3x3x3 neighbor cells around posK
    // and apply inline minimum-image subtraction.  Falls back to brute force if
    // the cell list is invalid (triclinic box, etc.).
    const bool useCellList = cellList_.valid;
    const real Lx2 = useCellList ? cellList_.Lx * 0.5_real : 0.0_real;
    const real Ly2 = useCellList ? cellList_.Ly * 0.5_real : 0.0_real;
    const real Lz2 = useCellList ? cellList_.Lz * 0.5_real : 0.0_real;

    auto sweepEnv = [&](const RVec& posK, int typeOld, int typeNew)
    {
        if (typeOld == typeNew) return;

        if (useCellList)
        {
            // Locate cell of posK
            real fx = posK[XX] * cellList_.invLx; fx -= std::floor(fx);
            real fy = posK[YY] * cellList_.invLy; fy -= std::floor(fy);
            real fz = posK[ZZ] * cellList_.invLz; fz -= std::floor(fz);
            const int ix = std::min(cellList_.nx - 1, std::max(0, int(fx * cellList_.nx)));
            const int iy = std::min(cellList_.ny - 1, std::max(0, int(fy * cellList_.ny)));
            const int iz = std::min(cellList_.nz - 1, std::max(0, int(fz * cellList_.nz)));

            for (int dzc = -1; dzc <= 1; dzc++)
                for (int dyc = -1; dyc <= 1; dyc++)
                    for (int dxc = -1; dxc <= 1; dxc++)
                    {
                        const int c = cellList_.cellIndex(ix + dxc, iy + dyc, iz + dzc);
                        for (int j : cellList_.cells[c])
                        {
                            if (inSwap(j)) continue;

                            // Inline orthorhombic minimum-image
                            real dxx = posK[XX] - xVec[j][XX];
                            real dxy = posK[YY] - xVec[j][YY];
                            real dxz = posK[ZZ] - xVec[j][ZZ];
                            if (dxx >  Lx2) dxx -= cellList_.Lx;
                            else if (dxx < -Lx2) dxx += cellList_.Lx;
                            if (dxy >  Ly2) dxy -= cellList_.Ly;
                            else if (dxy < -Ly2) dxy += cellList_.Ly;
                            if (dxz >  Lz2) dxz -= cellList_.Lz;
                            else if (dxz < -Lz2) dxz += cellList_.Lz;

                            const real r2 = dxx * dxx + dxy * dxy + dxz * dxz;
                            if (r2 > rvdw2 || r2 < 1.0e-10_real) continue;

                            const int jt = atype[j];
                            dE += ljEnergy(typeNew, jt, r2) - ljEnergy(typeOld, jt, r2);
                        }
                    }
        }
        else
        {
            // Brute-force fallback (triclinic box or cell list not built)
            rvec dx;
            for (int j = 0; j < nAtoms; j++)
            {
                if (inSwap(j)) continue;
                pbc_dx(&pbc, posK, xVec[j], dx);
                const real r2 = dx[XX] * dx[XX] + dx[YY] * dx[YY] + dx[ZZ] * dx[ZZ];
                if (r2 > rvdw2 || r2 < 1.0e-10_real) continue;
                const int jt = atype[j];
                dE += ljEnergy(typeNew, jt, r2) - ljEnergy(typeOld, jt, r2);
            }
        }
    };

    for (int p : c_diffPos)
    {
        const int tD = dppcBeadAtype_[p];
        const int tI = dipcBeadAtype_[p];
        sweepEnv(xVec[dppc.atomIndices[p]], tD, tI);
        sweepEnv(xVec[dipc.atomIndices[p]], tI, tD);
    }

    // -------- (2) Intra-residue G96 angle delta (4 angles) --------
    auto cosAngle = [&](const RVec& a, const RVec& b, const RVec& c)
    {
        rvec ba, bc;
        pbc_dx(&pbc, a, b, ba);
        pbc_dx(&pbc, c, b, bc);
        const real n2_ba = ba[XX] * ba[XX] + ba[YY] * ba[YY] + ba[ZZ] * ba[ZZ];
        const real n2_bc = bc[XX] * bc[XX] + bc[YY] * bc[YY] + bc[ZZ] * bc[ZZ];
        if (n2_ba < 1.0e-12_real || n2_bc < 1.0e-12_real) return 1.0_real;
        const real dot = ba[XX] * bc[XX] + ba[YY] * bc[YY] + ba[ZZ] * bc[ZZ];
        real c0  = dot / std::sqrt(n2_ba * n2_bc);
        if (c0 >  1.0_real) c0 =  1.0_real;
        if (c0 < -1.0_real) c0 = -1.0_real;
        return c0;
    };
    auto angE = [](real cosT, real cos0, real kc)
    {
        const real d = cosT - cos0;
        return 0.5_real * kc * d * d;
    };

    for (const auto& A : c_diffAngles)
    {
        const real cos0Dppc = std::cos(A.theta0Dppc * kDeg2Rad);
        const real cos0Dipc = std::cos(A.theta0Dipc * kDeg2Rad);
        const real cD = cosAngle(xVec[dppc.atomIndices[A.i]],
                                 xVec[dppc.atomIndices[A.j]],
                                 xVec[dppc.atomIndices[A.k]]);
        const real cI = cosAngle(xVec[dipc.atomIndices[A.i]],
                                 xVec[dipc.atomIndices[A.j]],
                                 xVec[dipc.atomIndices[A.k]]);
        // ΔE_intra = (E_DPPC@I + E_DIPC@D) - (E_DPPC@D + E_DIPC@I)
        dE += angE(cI, cos0Dppc, A.kcDppc) - angE(cD, cos0Dppc, A.kcDppc)
            + angE(cD, cos0Dipc, A.kcDipc) - angE(cI, cos0Dipc, A.kcDipc);
    }

    // -------- (3) Intra-residue LJ delta (pairs involving differing atoms, non-bonded) --------
    for (int ki = 0; ki < 12; ki++)
    {
        for (int kj = ki + 1; kj < 12; kj++)
        {
            if (isBondedPair(ki, kj)) continue;
            if (!isDiffPos(ki) && !isDiffPos(kj)) continue;

            rvec dxD, dxI;
            pbc_dx(&pbc, xVec[dppc.atomIndices[ki]], xVec[dppc.atomIndices[kj]], dxD);
            pbc_dx(&pbc, xVec[dipc.atomIndices[ki]], xVec[dipc.atomIndices[kj]], dxI);
            const real r2D = dxD[XX] * dxD[XX] + dxD[YY] * dxD[YY] + dxD[ZZ] * dxD[ZZ];
            const real r2I = dxI[XX] * dxI[XX] + dxI[YY] * dxI[YY] + dxI[ZZ] * dxI[ZZ];

            const int tDppc_i = dppcBeadAtype_[ki];
            const int tDppc_j = dppcBeadAtype_[kj];
            const int tDipc_i = dipcBeadAtype_[ki];
            const int tDipc_j = dipcBeadAtype_[kj];

            auto safeLJ = [&](int ai, int aj, real r2)
            { return (r2 > rvdw2 || r2 < 1.0e-10_real) ? 0.0_real : ljEnergy(ai, aj, r2); };

            const real eDppcAtI = safeLJ(tDppc_i, tDppc_j, r2I);
            const real eDppcAtD = safeLJ(tDppc_i, tDppc_j, r2D);
            const real eDipcAtI = safeLJ(tDipc_i, tDipc_j, r2I);
            const real eDipcAtD = safeLJ(tDipc_i, tDipc_j, r2D);

            dE += (eDppcAtI - eDppcAtD) + (eDipcAtD - eDipcAtI);
        }
    }

    // -------- (4) Cross LJ delta between the two swapped residues --------
    // LJ between the two swapped lipids themselves.  Nonzero only when they are
    // within the LJ cut-off of each other; skipped by (1) through inSwap().
    for (int bi = 0; bi < 12; bi++)
    {
        const int tD_i = dppcBeadAtype_[bi];   // type at x_bi before the swap
        const int tI_i = dipcBeadAtype_[bi];   // type at x_bi after  the swap

        for (int bj = 0; bj < 12; bj++)
        {
            const int tD_j = dppcBeadAtype_[bj];   // type at y_bj after  the swap
            const int tI_j = dipcBeadAtype_[bj];   // type at y_bj before the swap

            if (tD_i == tI_i && tD_j == tI_j) continue;

            rvec dx;
            pbc_dx(&pbc, xVec[dppc.atomIndices[bi]], xVec[dipc.atomIndices[bj]], dx);
            const real r2 = dx[XX] * dx[XX] + dx[YY] * dx[YY] + dx[ZZ] * dx[ZZ];
            if (r2 > rvdw2 || r2 < 1.0e-10_real) continue;

            dE += ljEnergy(tI_i, tD_j, r2) - ljEnergy(tD_i, tI_j, r2);
        }
    }

    return dE;
}

void LipidSwapMC::classifyLeaflets()
{
    auto& xVec = state_->x;
    RVec  vscenter = { 0, 0, 0 };
    int   tailCount = 0;

    auto addTail = [&](const ResidueInfo& ri)
    {
        if (ri.c4aIdx >= 0) { vscenter += xVec[ri.c4aIdx]; ++tailCount; }
        if (ri.c4bIdx >= 0) { vscenter += xVec[ri.c4bIdx]; ++tailCount; }
    };
    for (const auto& r : dppc_residues_) addTail(r);
    for (const auto& r : dipc_residues_) addTail(r);

    if (tailCount == 0) return;
    svmul(1.0 / tailCount, vscenter, vscenter);

    real avgTailDist = 0;
    auto sumDist = [&](const ResidueInfo& ri)
    {
        if (ri.c4aIdx >= 0) avgTailDist += norm(xVec[ri.c4aIdx] - vscenter);
        if (ri.c4bIdx >= 0) avgTailDist += norm(xVec[ri.c4bIdx] - vscenter);
    };
    for (const auto& r : dppc_residues_) sumDist(r);
    for (const auto& r : dipc_residues_) sumDist(r);
    avgTailDist /= tailCount;

    inner_dppc_.clear(); outer_dppc_.clear();
    inner_dipc_.clear(); outer_dipc_.clear();

    auto classify = [&](const std::vector<ResidueInfo>& residues,
                        std::vector<int>&               inner,
                        std::vector<int>&               outer)
    {
        for (int i = 0, n = static_cast<int>(residues.size()); i < n; i++)
        {
            if (residues[i].po4Idx < 0) continue;
            real dist = norm(xVec[residues[i].po4Idx] - vscenter);
            (dist < avgTailDist ? inner : outer).push_back(i);
        }
    };
    classify(dppc_residues_, inner_dppc_, outer_dppc_);
    classify(dipc_residues_, inner_dipc_, outer_dipc_);
}

real LipidSwapMC::computeEnergy(int64_t step, double time)
{
    GMX_ASSERT(localTopology_ != nullptr, "Local topology not set");

    // Construct virtual sites for the (possibly swapped) coordinates BEFORE
    // do_force, exactly as legacy do_md does at the top of every MD step.
    if (vsite_ != nullptr)
    {
        vsite_->construct(state_->x.arrayRefWithPadding().unpaddedArrayRef(),
                          state_->v.arrayRefWithPadding().unpaddedArrayRef(),
                          state_->box,
                          VSiteOperation::Positions);
    }

    // do_force() takes its workload from runScheduleWork_->stepWork, i.e.
    // from the enclosing MD step.  The pair list is rebuilt only on a
    // neighbor-search step (guaranteed by the nstlist check in the
    // constructor) and F_EPOT is updated only on an energy step.

    tensor force_vir;
    clear_mat(force_vir);

    const gmx_multisim_t* ms  = nullptr;
    gmx_edsam*            ed  = nullptr;
    Awh*                  awh = nullptr;

    // Use the legacy do_md ForceBuffers (already sized and pinned correctly)
    auto& fview = forceBuffers_->view();

    do_force(fplog_,
             cr_,
             ms,
             *inputrec_,
             mdModulesNotifiers_,
             awh,
             enforcedRotation_,
             imdSession_,
             pull_work_,
             step,
             nrnb_,
             wcycle_,
             localTopology_,
             state_->box,
             state_->x.arrayRefWithPadding(),
             state_->v.arrayRefWithPadding().unpaddedArrayRef(),
             nullptr,
             &fview,
             force_vir,
             mdAtoms_->mdatoms(),
             enerd_,
             state_->lambda,
             fr_,
             *runScheduleWork_,
             vsite_,
             nullptr,   // mu_tot
             time,
             ed,
             fr_->longRangeNonbondeds.get(),
             DDBalanceRegionHandler(cr_));

    return enerd_->term[F_EPOT];
}

void LipidSwapMC::apply(int64_t step, double time)
{
    // Cholesterol virtual sites were constructed at the top of this MD step and
    // therefore lag the post-update coordinates.  Refresh them so that
    // computeSwapDeltaE() sees the same environment as computeEnergy() does.
    if (vsite_ != nullptr)
    {
        vsite_->construct(state_->x.arrayRefWithPadding().unpaddedArrayRef(),
                          state_->v.arrayRefWithPadding().unpaddedArrayRef(),
                          state_->box,
                          VSiteOperation::Positions);
    }

    if (inner_dppc_.empty() || outer_dipc_.empty()
        || inner_dipc_.empty() || outer_dppc_.empty())
    {
        if (fplog_)
        {
            fprintf(fplog_,
                    "LipidSwapMC step %lld: skipping - one leaflet pool is empty.\n",
                    (long long)step);
        }
        return;
    }

    auto&      xVec   = state_->x;
    int        rule   = -1;
    bool       accept   = false;
    bool       anyAccepted = false;
    bool       forceBufferDirty = false;  // set whenever computeEnergy was called with a non-current state

    // Build cell list once per MC block (orthorhombic only).
    buildCellList();

    // Per-block statistics accumulator.
    BlockStats bs;

    for (int mc = 0; mc < mc_nsteps_; mc++)
    {
        if (inner_dppc_.empty() || outer_dipc_.empty()
            || inner_dipc_.empty() || outer_dppc_.empty()) break;

        const int n1   = static_cast<int>(inner_dppc_.size() * outer_dipc_.size());
        const int n2   = static_cast<int>(inner_dipc_.size() * outer_dppc_.size());
        const int ntot = n1 + n2;
        if (ntot == 0) break;

        int dppcListIdx = 0;
        int dipcListIdx = 0;

        if (uniform01_(rng_) < static_cast<double>(n1) / ntot)
        {
            rule = 0;
            dppcListIdx = inner_dppc_[
                static_cast<size_t>(uniform01_(rng_) * inner_dppc_.size()) % inner_dppc_.size()];
            dipcListIdx = outer_dipc_[
                static_cast<size_t>(uniform01_(rng_) * outer_dipc_.size()) % outer_dipc_.size()];
        }
        else
        {
            rule = 1;
            dipcListIdx = inner_dipc_[
                static_cast<size_t>(uniform01_(rng_) * inner_dipc_.size()) % inner_dipc_.size()];
            dppcListIdx = outer_dppc_[
                static_cast<size_t>(uniform01_(rng_) * outer_dppc_.size()) % outer_dppc_.size()];
        }

        ResidueInfo& dppcRes = dppc_residues_[dppcListIdx];
        ResidueInfo& dipcRes = dipc_residues_[dipcListIdx];

        if (dppcRes.atomIndices.size() != dipcRes.atomIndices.size()) continue;
        const int nBeads = static_cast<int>(dppcRes.atomIndices.size());

        // Local energy difference, no coordinate change yet.
        const real dE = computeSwapDeltaE(dppcRes, dipcRes);

        // Validation needs F_EPOT, which do_force() updates only on energy steps.
        if (validateDeltaE_ && total_ < 5 && runScheduleWork_->stepWork.computeEnergy)
        {
            // One-shot validation: compare against full do_force ΔE.
            const real eBefore = computeEnergy(step, time);
            std::vector<RVec> savedDppc(nBeads), savedDipc(nBeads);
            for (int b = 0; b < nBeads; b++)
            {
                savedDppc[b] = xVec[dppcRes.atomIndices[b]];
                savedDipc[b] = xVec[dipcRes.atomIndices[b]];
                xVec[dppcRes.atomIndices[b]] = savedDipc[b];
                xVec[dipcRes.atomIndices[b]] = savedDppc[b];
            }
            const real eAfter = computeEnergy(step, time);
            const real dE_full = eAfter - eBefore;
            // Restore to pre-swap state for the actual decision below.
            for (int b = 0; b < nBeads; b++)
            {
                xVec[dppcRes.atomIndices[b]] = savedDppc[b];
                xVec[dipcRes.atomIndices[b]] = savedDipc[b];
            }
            if (fplog_)
            {
                fprintf(fplog_,
                        "LipidSwapMC validate: step %lld attempt %lld   "
                        "dE_local = %.4f   dE_full = %.4f   diff = %.4f kJ/mol\n",
                        (long long)step, (long long)total_, dE, dE_full, dE - dE_full);
            }
            // computeEnergy() was called with swapped coords above.  After
            // restoring, the force buffer is now stale relative to the
            // current (pre-swap) coords, so flag a refresh at end of block.
            forceBufferDirty = true;
        }

        // Metropolis-Hastings: g(reverse)/g(forward) = ntot_before / ntot_after.
        // The proposal samples uniformly across all (n1+n2) valid swap pairs;
        // after a swap the pool counts change by ±1 so ntot shifts slightly.
        const int nIDp = static_cast<int>(inner_dppc_.size());
        const int nODp = static_cast<int>(outer_dppc_.size());
        const int nIDi = static_cast<int>(inner_dipc_.size());
        const int nODi = static_cast<int>(outer_dipc_.size());
        int n1_after, n2_after;
        if (rule == 0)
        {
            n1_after = (nIDp - 1) * (nODi - 1);
            n2_after = (nIDi + 1) * (nODp + 1);
        }
        else
        {
            n1_after = (nIDp + 1) * (nODi + 1);
            n2_after = (nIDi - 1) * (nODp - 1);
        }
        const int    ntot_after = n1_after + n2_after;
        const double mhRatio    = (ntot_after > 0)
                ? static_cast<double>(ntot) / static_cast<double>(ntot_after)
                : 1.0;
        const double accProb = std::min(1.0, std::exp(-beta_ * dE) * mhRatio);
        accept               = (uniform01_(rng_) < accProb);

        // -------- statistics tracking (block-local) --------
        const double dEd = static_cast<double>(dE);
        if (!bs.haveAny)
        {
            bs.dE_min = bs.dE_max = dEd;
            bs.haveAny = true;
        }
        else
        {
            if (dEd < bs.dE_min) bs.dE_min = dEd;
            if (dEd > bs.dE_max) bs.dE_max = dEd;
        }
        bs.dE_sum += dEd;
        bs.mh_sum += mhRatio;
        bs.att++;

        if (accept)
        {
            // Visualization-only: BEFORE frame -- the two candidate lipids are
            // flagged (swap=1) at their pre-swap positions, everything else 0.
            if (swapTrajEnabled_)
            {
                writeSwapFrame(dppcRes, dipcRes);
            }

            // Apply the swap to the actual state coordinates.
            for (int b = 0; b < nBeads; b++)
            {
                std::swap(xVec[dppcRes.atomIndices[b]], xVec[dipcRes.atomIndices[b]]);
            }
            // Update cell-list membership of the 24 moved atoms.
            for (int b = 0; b < nBeads; b++)
            {
                updateCellListAtom(dppcRes.atomIndices[b]);
                updateCellListAtom(dipcRes.atomIndices[b]);
            }
            anyAccepted = true;
            accepted_++;

            bs.acc++;
            bs.dE_acc_sum += dEd;

            if (rule == 0)
            {
                inner_dppc_.erase(std::find(inner_dppc_.begin(), inner_dppc_.end(), dppcListIdx));
                outer_dppc_.push_back(dppcListIdx);
                outer_dipc_.erase(std::find(outer_dipc_.begin(), outer_dipc_.end(), dipcListIdx));
                inner_dipc_.push_back(dipcListIdx);
                rule0_accepted_++;
                bs.r0++;
            }
            else
            {
                inner_dipc_.erase(std::find(inner_dipc_.begin(), inner_dipc_.end(), dipcListIdx));
                outer_dipc_.push_back(dipcListIdx);
                outer_dppc_.erase(std::find(outer_dppc_.begin(), outer_dppc_.end(), dppcListIdx));
                inner_dppc_.push_back(dppcListIdx);
                rule1_accepted_++;
                bs.r1++;
            }

            // Visualization-only: AFTER frame -- same two lipids flagged, now
            // at their swapped positions.  BEFORE->AFTER shows the swap motion.
            if (swapTrajEnabled_)
            {
                writeSwapFrame(dppcRes, dipcRes);
            }
        }

        total_++;
    }

    // One block-summary line.
    if (mcLogFile_)
    {
        const int blockIdx = (mc_freq_ > 0) ? static_cast<int>(step / mc_freq_) : 0;
        writeBlockLog(step, time, blockIdx, bs);
    }

    // Refresh neighbor-list / force-kernel state so the next MD step sees a
    // consistent NS/topology mapping.  This is a neighbor-search step (see the
    // nstlist check in the constructor), so do_force() rebuilds the pair list.
    // Required when:
    //   - any swap was accepted (positions changed) OR
    //   - the force buffer is dirty (computeEnergy was called with a state
    //     that differs from the current coords, i.e. validation).
    if (anyAccepted || forceBufferDirty)
    {
        computeEnergy(step, time);
    }
}

void LipidSwapMC::openLogFile()
{
    mcLogFile_ = std::fopen("swap.log", "a");
    if (!mcLogFile_ && fplog_)
    {
        fprintf(fplog_, "Warning: LipidSwapMC cannot open swap.log for writing.\n");
    }
}

void LipidSwapMC::buildVizAtoms(const gmx_mtop_t& top)
{
    // Collect global atom indices of all DPPC/DIPC/CHOL atoms, with a type code
    // (1=DPPC, 2=DIPC, 3=CHOL) for OVITO coloring.  Water/ions are excluded.
    vizAtoms_.clear();
    vizType_.clear();

    int globalAtomOffset = 0;
    for (const gmx_molblock_t& molblock : top.molblock)
    {
        const gmx_moltype_t& moltype   = top.moltype[molblock.type];
        const t_atoms&       atoms     = moltype.atoms;
        const int            nAtomsMol = atoms.nr;

        for (int mol = 0; mol < molblock.nmol; mol++)
        {
            const int molOffset = globalAtomOffset + mol * nAtomsMol;
            for (int a = 0; a < nAtomsMol; a++)
            {
                const char* rname = *(atoms.resinfo[atoms.atom[a].resind].name);
                int         type  = 0;
                if (std::strcmp(rname, "DPPC") == 0) type = 1;
                else if (std::strcmp(rname, "DIPC") == 0) type = 2;
                else if (std::strcmp(rname, "CHOL") == 0) type = 3;
                if (type > 0)
                {
                    vizAtoms_.push_back(molOffset + a);
                    vizType_.push_back(type);
                }
            }
        }
        globalAtomOffset += molblock.nmol * nAtomsMol;
    }

    if (fplog_)
    {
        fprintf(fplog_,
                "LipidSwapMC viz: dumping %zu lipid atoms (DPPC/DIPC/CHOL) per swap frame.\n",
                vizAtoms_.size());
    }
}

void LipidSwapMC::openSwapTraj()
{
    // Truncate-and-write ("w"): each run starts a fresh visualization trajectory.
    swapTrajFile_ = std::fopen("mc_swaps.lammpstrj", "w");
    if (!swapTrajFile_ && fplog_)
    {
        fprintf(fplog_, "Warning: LipidSwapMC cannot open mc_swaps.lammpstrj for writing.\n");
    }
}

void LipidSwapMC::writeSwapFrame(const ResidueInfo& dppc, const ResidueInfo& dipc)
{
    if (!swapTrajFile_) return;

    const auto& xVec = state_->x;
    const matrix& box = state_->box;

    // The 24 atoms involved in this swap get swap=1; everyone else 0.
    // Use a small flat set for O(1)-ish membership over the lipid subset.
    std::array<int, 24> swapped{};
    const int nB = static_cast<int>(dppc.atomIndices.size());
    for (int b = 0; b < nB && b < 12; b++)
    {
        swapped[b]      = dppc.atomIndices[b];
        swapped[b + 12] = dipc.atomIndices[b];
    }
    std::sort(swapped.begin(), swapped.end());
    auto isSwapped = [&](int g)
    { return std::binary_search(swapped.begin(), swapped.end(), g); };

    const int natoms = static_cast<int>(vizAtoms_.size());

    // LAMMPS custom dump header (orthorhombic box; coords in nm).
    std::fprintf(swapTrajFile_, "ITEM: TIMESTEP\n%lld\n", (long long)swapFrameIdx_);
    std::fprintf(swapTrajFile_, "ITEM: NUMBER OF ATOMS\n%d\n", natoms);
    std::fprintf(swapTrajFile_, "ITEM: BOX BOUNDS pp pp pp\n");
    std::fprintf(swapTrajFile_, "0 %.5f\n0 %.5f\n0 %.5f\n",
                 box[XX][XX], box[YY][YY], box[ZZ][ZZ]);
    std::fprintf(swapTrajFile_, "ITEM: ATOMS id type x y z swap\n");

    for (int k = 0; k < natoms; k++)
    {
        const int  g   = vizAtoms_[k];
        const int  ty  = vizType_[k];
        const int  flg = isSwapped(g) ? 1 : 0;
        std::fprintf(swapTrajFile_, "%d %d %.4f %.4f %.4f %d\n",
                     g + 1, ty, xVec[g][XX], xVec[g][YY], xVec[g][ZZ], flg);
    }

    ++swapFrameIdx_;
}

void LipidSwapMC::writeBlockLog(int64_t           step,
                                double            time,
                                int               blockIdx,
                                const BlockStats& bs)
{
    if (!mcLogFile_) return;

    if (!headerWritten_)
    {
        fprintf(mcLogFile_,
                "# Lipid Swap MC per-block summary (one line per MC block).\n"
                "# step       MD step at end of block\n"
                "# block      block index (1-based)\n"
                "# time_ps    simulation time at end of block (ps)\n"
                "# att        attempts in this block (= lipidswap-mc-nstep unless a pool ran empty)\n"
                "# acc        accepted in this block\n"
                "# ratio      acc / att (this block)\n"
                "# acc_cum    cumulative accepted\n"
                "# ratio_cum  cumulative acc / att\n"
                "# r0, r1     rule-0 / rule-1 accepted in this block\n"
                "# r0_cum, r1_cum  cumulative rule counts\n"
                "# in/out_DPPC, in/out_DIPC  leaflet populations\n"
                "# dE_mean    mean dE over all attempts in block (kJ/mol)\n"
                "# dE_min     most negative dE in block\n"
                "# dE_max     most positive dE in block\n"
                "# dE_acc     mean dE of accepted attempts only\n"
                "# mh_mean    mean Metropolis-Hastings ratio (~1 by design)\n");
        fprintf(mcLogFile_,
                "# %8s  %5s  %11s  %5s  %5s  %8s  %8s  %9s  %4s  %4s  "
                "%7s  %7s  %7s  %8s  %7s  %8s  "
                "%12s  %12s  %12s  %12s  %10s\n",
                "step", "block", "time_ps", "att", "acc",
                "ratio", "acc_cum", "ratio_cum", "r0", "r1",
                "r0_cum", "r1_cum",
                "in_DPPC", "out_DPPC", "in_DIPC", "out_DIPC",
                "dE_mean", "dE_min", "dE_max", "dE_acc", "mh_mean");
        headerWritten_ = true;
    }

    const double ratio_block = (bs.att > 0)
            ? static_cast<double>(bs.acc) / bs.att : 0.0;
    const double ratio_cum   = (total_ > 0)
            ? static_cast<double>(accepted_) / total_ : 0.0;
    const double dE_mean     = (bs.att > 0) ? bs.dE_sum / bs.att : 0.0;
    const double dE_acc_mean = (bs.acc > 0) ? bs.dE_acc_sum / bs.acc : 0.0;
    const double mh_mean     = (bs.att > 0) ? bs.mh_sum / bs.att : 1.0;
    const double dE_min      = bs.haveAny ? bs.dE_min : 0.0;
    const double dE_max      = bs.haveAny ? bs.dE_max : 0.0;

    fprintf(mcLogFile_,
            "%10lld  %5d  %11.3f  %5d  %5d  %8.6f  %8lld  %9.6f  %4d  %4d  "
            "%7lld  %7lld  %7zu  %8zu  %7zu  %8zu  "
            "%12.4f  %12.4f  %12.4f  %12.4f  %10.6f\n",
            (long long)step,
            blockIdx,
            time,
            bs.att,
            bs.acc,
            ratio_block,
            (long long)accepted_,
            ratio_cum,
            bs.r0,
            bs.r1,
            (long long)rule0_accepted_,
            (long long)rule1_accepted_,
            inner_dppc_.size(),
            outer_dppc_.size(),
            inner_dipc_.size(),
            outer_dipc_.size(),
            dE_mean,
            dE_min,
            dE_max,
            dE_acc_mean,
            mh_mean);
    std::fflush(mcLogFile_);
}

} // namespace gmx
