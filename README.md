# Lipid Swap Monte Carlo for GROMACS 2025.2

A hybrid Monte Carlo / molecular dynamics (MC-MD) extension of GROMACS 2025.2
for equilibrating the leaflet composition of MARTINI lipid vesicles.

In a self-assembled or pre-built vesicle, the number of each lipid species in
the inner and outer leaflets is fixed by the construction, and spontaneous
flip-flop is far too slow to correct it on MD time scales. This extension adds
MC moves that exchange a DPPC in one leaflet with a DIPC in the other leaflet.
Each exchange is accepted or rejected with the Metropolis criterion, so the
transbilayer distribution relaxes toward equilibrium while the MD run
continues.

The MC moves run inside mdrun's MD loop: no external driver, file I/O or
restart is needed between MD and MC. The energy change of an exchange is
computed locally from the interactions that the exchange changes.

This is not an official GROMACS release. It is the official GROMACS 2025.2
source with Lipid Swap MC added.

## Supported systems

- **Lipids:** 12-bead MARTINI 2 DPPC and DIPC in a vesicle.
- **Interactions:** LJ cut-off with potential-shift (or no modifier);
  tested with reaction-field electrostatics.
- **Integrator:** `md` or `sd`, with one temperature for all coupling groups.
- **mdrun:** a single rank (`-ntmpi 1`) and CPU update (`-update cpu`).
- **MC interval:** `lipidswap-mc-nst` a multiple of 400
  (see [Choosing lipidswap-mc-nst](#choosing-lipidswap-mc-nst)).

mdrun checks these at start-up and stops with an explanation if one is not met.

## Building

Build it like any GROMACS 2025.2 source tree; the
[GROMACS installation guide](https://manual.gromacs.org/2025.2/install-guide/index.html)
applies unchanged. Requirements: CMake 3.28 or newer, a C++17 compiler and,
for GPU runs, CUDA 12.1 or newer.

Get the source in one of two ways.

1. Clone this repository:

   ```bash
   git clone https://github.com/woosi2/lipidswap-gromacs.git
   cd lipidswap-gromacs
   ```

2. Or apply the patch from the release page to the official source:

   ```bash
   wget https://ftp.gromacs.org/gromacs/gromacs-2025.2.tar.gz
   md5sum gromacs-2025.2.tar.gz    # d7262d5e2a51995735860e71654629b2
   tar xzf gromacs-2025.2.tar.gz && cd gromacs-2025.2
   patch -p1 < lipidswap-gromacs-2025.2.patch
   ```

Then configure, build and install:

```bash
mkdir build && cd build
cmake .. -DGMX_GPU=CUDA -DGMX_BUILD_OWN_FFTW=ON \
         -DCMAKE_INSTALL_PREFIX=$HOME/opt/gromacs-lipidswap
make -j 16 && make install
source $HOME/opt/gromacs-lipidswap/bin/GMXRC
```

Use this build for both `gmx grompp` and `gmx mdrun`. Its run input files
carry the tpx tag `lipidswap`: official GROMACS cannot run them, although its
analysis tools can still read their topology. Run input files from official
GROMACS 2025.2 are read normally, with lipid swap MC off.

## Usage

### mdp options

Add these lines to a normal MD mdp file:

```
lipidswap-mc       = yes     ; enable lipid swap MC (default: no)
lipidswap-mc-nst   = 50000   ; MD steps between MC blocks (default: 1000)
lipidswap-mc-nstep = 5000    ; swap attempts per MC block (default: 1000)
lipidswap-mc-seed  = -1      ; random seed; -1 draws one (default: -1)
```

The MC temperature is the `ref-t` of the temperature-coupling groups.
With `lipidswap-mc-seed = -1` the seed that was drawn is written to the log
file, so a run can be repeated by setting that value.

### Choosing lipidswap-mc-nst

MC blocks must fall on neighbor-search steps, so `lipidswap-mc-nst` must be a
multiple of the nstlist that mdrun actually uses. mdrun may increase nstlist
at start-up, on CPUs and GPUs alike, to 20, 25, 40, 50, 80 or 100. A multiple
of 400 (the least common multiple of these values) is therefore always safe.
Alternatively, fix nstlist with `mdrun -nstlist` and use a multiple of that
value.

### Starting structure

The leaflets are assigned once, at the start of mdrun, from the distance of
each PO4 bead to the vesicle centre. The vesicle must therefore be whole and
centred in the box in the starting structure. A vesicle split across the
periodic boundary can be reassembled and centred in two steps:

```bash
gmx select -s start.gro -on pbc.ndx -select '"membrane" resname DPPC DIPC CHOL' '"System" all'
printf "membrane\nSystem\n" | gmx trjconv -s topol.tpr -f start.gro -n pbc.ndx \
    -pbc cluster -o clustered.gro
printf "membrane\nSystem\n" | gmx trjconv -s topol.tpr -f clustered.gro -n pbc.ndx \
    -pbc mol -center -ur compact -o centered.gro
```

### Running

```bash
gmx grompp -f md.mdp -c centered.gro -p topol.top -n index.ndx -o md.tpr
gmx mdrun -deffnm md -ntmpi 1 -update cpu -nb gpu
```

Nonbonded and bonded forces may run on the GPU as usual.

### Environment variables

| Variable | Effect |
|---|---|
| `GMX_LIPIDSWAP_VALIDATE=1` | For the first 5 attempts, also compute the energy change from two full force evaluations and write both values to the log file. Use a double-precision build for a meaningful comparison. |
| `GMX_LIPIDSWAP_TRAJ=1` | Write each accepted swap to `mc_swaps.lammpstrj` (see [Output](#mc_swapslammpstrj)). |

## Output

### Log file (md.log)

At start-up:

```
LipidSwapMC local dE: ntype=8, rvdw=1.100 nm, modifier=PotShift.
  DPPC bead types: 0 1 2 2 3 3 3 3 3 3 3 3
  DIPC bead types: 0 1 2 2 3 4 4 3 3 4 4 3

Lipid Swap MC: found 1704 DPPC and 1704 DIPC residues.
  MC frequency: every 400 MD steps, 200 moves per block.
  Temperature : 303.15 K  (beta = 0.3967 mol/kJ)
  RNG seed    : 12345
  Swap traj   : off
```

At the end of a run that finishes normally:

```
Lipid Swap MC summary: 18 accepted / 1000 total (ratio 0.0180)
  Rule-0 (DPPC-in <-> DIPC-out): 6 accepted
  Rule-1 (DIPC-in <-> DPPC-out): 12 accepted
```

### swap.log

One line per MC block, written to the working directory and flushed after
each block, so the file is complete even if the run is killed. A restart
appends to the same file. Lines starting with `#` are the header; columns
are separated by whitespace.

| Columns | Meaning |
|---|---|
| `step`, `block`, `time_ps` | MD step, block index and time of the MC block |
| `att`, `acc`, `ratio` | attempts, accepted swaps and acceptance ratio in this block |
| `acc_cum`, `ratio_cum` | accepted swaps and acceptance ratio so far in this mdrun |
| `r0`, `r1`, `r0_cum`, `r1_cum` | accepted rule-0 / rule-1 swaps, in this block and so far |
| `in_DPPC`, `out_DPPC`, `in_DIPC`, `out_DIPC` | leaflet populations after the block |
| `dE_mean`, `dE_min`, `dE_max`, `dE_acc` | energy changes in this block (kJ/mol); `dE_acc` over accepted swaps only |
| `mh_mean` | mean Metropolis-Hastings proposal ratio (close to 1) |

### mc_swaps.lammpstrj

Written only with `GMX_LIPIDSWAP_TRAJ=1`: two frames per accepted swap (before
and after), with columns `id type x y z swap`. `type` is 1 for DPPC, 2 for DIPC
and 3 for CHOL; `swap = 1` marks the 24 beads of the exchanged pair. The file
is overwritten when mdrun starts.

## Method

Every `lipidswap-mc-nst` MD steps, at the end of the step, mdrun runs one MC
block of `lipidswap-mc-nstep` attempts. Each attempt:

1. **Picks a pair** uniformly: an inner DPPC with an outer DIPC, or an inner
   DIPC with an outer DPPC.
2. **Computes the energy change ΔE locally.** DPPC and DIPC differ only in the
   atom types of four tail beads and in four tail angles, so ΔE is the sum of
   - LJ between the changed beads and their surroundings,
   - the four G96 angle terms,
   - LJ within each of the two lipids,
   - LJ between the two lipids (nonzero only within the cut-off).
3. **Accepts** with probability `min(1, exp(−ΔE/kT) · N/N′)`, where `N` and
   `N′` are the numbers of possible pairs before and after the exchange.
4. **Swaps the coordinates** of the two lipids on acceptance (velocities are
   kept) and updates the leaflet lists.

After the block the pair list is rebuilt, so the MD continues with the
exchanged lipids in place.

## Notes

- **Restarts.** When a run is continued from a checkpoint (`mdrun -cpi`),
  the leaflets are reassigned from the checkpoint coordinates and the MC
  counters start again from zero: the cumulative columns of `swap.log` and
  the summary in the log file cover only the current mdrun. `swap.log` is
  appended to, with a second header. With a fixed `lipidswap-mc-seed` the
  same random sequence is used again, so use `-1` or a new seed for each
  continuation.
- **Leaflet lists.** After start-up, the leaflet lists change only through
  accepted swaps. A lipid that moves to the other leaflet during MD, by
  spontaneous flip-flop or through a pore, is still counted in its original
  leaflet. This is rare for MARTINI DPPC and DIPC, but in systems with pores
  or in very long runs, restart mdrun from time to time to reassign the
  leaflets.

## Planned extensions

The current version is limited to DPPC and DIPC. A future version will read
the differing atom types and bonded parameters from the topology instead of
assuming the MARTINI 2 DPPC/DIPC layout, so that any two lipids with the same
topology (the same beads, bonds and exclusions) can be exchanged, including
MARTINI 3 lipids.

## Contact

Questions, bug reports and discussion are always welcome. Please open an
issue on GitHub or write to Wooseok Jung (wshr0307@sogang.ac.kr).

## License

GROMACS and this extension are distributed under the GNU Lesser General
Public License, version 2.1 or later; see [COPYING](COPYING).

## Citation

A paper describing this method is in preparation. Until then, please cite
this repository and GROMACS
([Abraham et al., SoftwareX 2015](https://doi.org/10.1016/j.softx.2015.06.001)).
