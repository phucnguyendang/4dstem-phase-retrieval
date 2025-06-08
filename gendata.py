import numpy as np
import abtem
import ase
from ase import Atoms
import h5py
import random
import gc
import argparse
from dask.diagnostics import ProgressBar

# ======================== #
#       ARGUMENT PARSER   #
# ======================== #
def parse_args():
    parser = argparse.ArgumentParser(description="STEM Simulation with abTEM")

    parser.add_argument("--start_id", type=int, required=True, help="Start index of materials to simulate")
    parser.add_argument("--end_id", type=int, required=True, help="End index of materials to simulate")
    parser.add_argument("--cell_size", type=float, required=True, help="Size of the simulation cell in x and y (Angstrom)")
    parser.add_argument("--cell_depth", type=float, default=random.uniform(10, 20), help="Depth of the simulation cell in z (Angstrom)")
    parser.add_argument("--probe_energy", type=float, required=True, help="Probe energy in eV")
    parser.add_argument("--wave_function_size", type=int, required=True, help="Number of pixels in wave function")
    parser.add_argument("--semiangle_cutoff", type=float, required=True, help="Semi-angle cutoff in mrad")
    parser.add_argument("--defocus", type=float, required=True, help="Defocus value in Angstrom")
    parser.add_argument("--sampling_step", type=float, required=True, help="Scanning step size in Angstrom")
    parser.add_argument("--num_phonon_configs", type=int, required=True, help="Number of frozen phonon configurations")
    parser.add_argument("--spherical_aberration", type=float, required=True, help="Spherical aberration coefficient in Angstrom")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the HDF5 file with material data")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the HDF5 file to store output")

    return parser.parse_args()


# ======================== #
#    LOAD MATERIAL DATA    #
# ======================== #
def get_materials(hdf5_filename, start_id, end_id):
    samples = []
    with h5py.File(hdf5_filename, "r") as hdf5_file:
        for crystal_type in hdf5_file.keys():
            material_group = hdf5_file[crystal_type]["materials"]
            material_keys = list(material_group.keys())
            sampled_keys = material_keys[start_id:end_id]
            for key in sampled_keys:
                material = material_group[key]
                sample_data = {
                    "material_id": material["material_id"][()],
                    "cart_coords": np.array(material["cart_coords"]),
                    "lattice_matrix": np.array(material["lattice_matrix"]),
                    "pbc": np.array(material["pbc"]),
                    "atomic_numbers": np.array(material["atomic_numbers"])
                }
                samples.append(sample_data)
    return samples

# ======================== #
#         MAIN LOOP        #
# ======================== #
def main():
    args = parse_args()
    abtem.config.set({"device": "gpu"})

    sampled_data = get_materials(args.input_file, args.start_id, args.end_id)

    with h5py.File(args.output_file, "a") as f:
        for idx, sample in enumerate(sampled_data):
            print(f"Calculating material {idx} ...\n")

            specimen = Atoms(numbers=sample['atomic_numbers'],
                             positions=sample['cart_coords'],
                             cell=sample['lattice_matrix'],
                             pbc=sample['pbc'])

            specimen = abtem.atoms.cut_cell(specimen, cell=(args.cell_size, args.cell_size, args.cell_depth), margin=1)

            frozen_phonon = abtem.FrozenPhonons(
                specimen, num_configs=args.num_phonon_configs, sigmas=0.1, ensemble_mean=True
            )

            ideal_potential = abtem.Potential(
                atoms=specimen, gpts=args.wave_function_size,
                slice_thickness=frozen_phonon.cell[2][2],
                parametrization='kirkland', projection='infinite', device='gpu'
            )

            potential_numpy = ideal_potential.build().array.compute().get()
            f.create_dataset(f"phase_{idx}", data=potential_numpy, compression="gzip")

            potential = abtem.Potential(
                atoms=frozen_phonon, gpts=args.wave_function_size,
                slice_thickness=frozen_phonon.cell[2][2],
                parametrization='kirkland', projection='infinite', device='gpu'
            )

            probe = abtem.Probe(
                energy=args.probe_energy, semiangle_cutoff=args.semiangle_cutoff,
                defocus=args.defocus, device='gpu',
                aberrations={'Cs': args.spherical_aberration}
            )
            probe.match_grid(potential)

            pixelated_detector = abtem.PixelatedDetector(max_angle='full')
            grid_scan = abtem.GridScan(sampling=args.sampling_step)

            measurement = probe.scan(scan=grid_scan, detectors=pixelated_detector, potential=potential)
            print(measurement.shape)

            with ProgressBar():
                dp_set = measurement.array.compute()

            f.create_dataset(f"dp_set_{idx}", data=dp_set, compression="gzip")

            del potential_numpy, dp_set
            gc.collect()

# ======================== #
#   RUN MAIN IF EXECUTED   #
# ======================== #
if __name__ == "__main__":
    main()
