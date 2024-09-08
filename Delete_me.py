def generate_uvspec_aerosol_opac(self, wavelength, sza, umu, phi, streams=16, zout='TOA', solver='disort', phi_0=0,
                                 aerosol_species_file='continental_clean', brdf_dict=None,
                                 polarized=False, mc_photons=1000000):
    # This is the default value
    if brdf_dict is None:
        brdf_dict = {'type': 'lambertian', 'albedo': 0.2}

    now = datetime.now()
    date_str = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(os.path.join(self.auto_io_path, 'UVSPEC_AEROSOL_AUTO.INP'), 'w') as f:
        f.write(f"# {date_str}\n")
        f.write('atmosphere_file ../data/atmmod/afglus.dat\n')
        f.write(f'rte_solver {solver}        # Radiative transfer equation solver\n')
        if solver == 'polradtran' or solver == 'mystic':
            if polarized:
                f.write('polradtran nstokes 3\n')
            else:
                f.write('polradtran nstokes 1\n')
        if solver == 'mystic':
            f.write(f'mc_photons {mc_photons}\n')
            f.write('mc_vroom off \n')
        # Location of the extraterrestrial spectrum
        f.write('source solar ../data/solar_flux/atlas_plus_modtran\n')
        if brdf_dict['type'] == 'lambertian':
            f.write(f"albedo {brdf_dict['albedo']}\n")
        elif brdf_dict['type'] == 'rossli':
            f.write(f"brdf_ambrals iso {brdf_dict['iso']}\n")
            f.write(f"brdf_ambrals vol {brdf_dict['vol']}\n")
            f.write(f"brdf_ambrals geo {brdf_dict['geo']}\n")
        elif brdf_dict['type'] == 'rpv':
            f.write(f"brdf_rpv rho0  {brdf_dict['rho0']}\n")
            f.write(f"brdf_rpv k     {brdf_dict['k']}\n")
            f.write(f"brdf_rpv theta {brdf_dict['theta']}\n")
            f.write(f"brdf_rpv sigma {brdf_dict['sigma']}\n")
            f.write(f"brdf_rpv t1 {brdf_dict['t1']}\n")
            f.write(f"brdf_rpv t2 {brdf_dict['t2']}\n")
            f.write(f"brdf_rpv scale {brdf_dict['scale']}\n")

        f.write('sza ' + str(sza) + '\n')  # Specify gamma size distribution (alpha=7)
        f.write(f'phi0 {phi_0}                  # Solar azimuth angle\n')
        f.write('phi ' + array_to_str(phi) + '\n')  # Calculate all phase matrix elements
        f.write('umu ' + array_to_str(umu) + '\n')
        f.write('zout ' + str(zout) + ' \n')
        f.write('wavelength ' + str(wavelength) + '\n')  # Specify effective radius grid
        f.write(f"number_of_streams  {round(streams)}     # Number of streams\n")
        f.write('aerosol_default  # switch on aerosol\n')
        f.write(f"aerosol_species_file {aerosol_species_file}\n")
        # f.write('no_scattering mol\n')
        # f.write('no_absorption\n')
        f.write(" print_disort_info 1 3 5\n")
        f.write('quiet\n')


def generate_uvspec_cloud(self, wavelength, sza, umu, phi, streams=16, zout='TOA', solver='disort', phi_0=0,
                          cod=15, brdf_dict=None, polarized=False, mc_photons=1000000):
    os.chdir(self.path + '/auto_io_files')
    now = datetime.now()
    date_str = now.strftime("%m/%d/%Y, %H:%M:%S")
    # This is the default value
    if brdf_dict is None:
        brdf_dict = {'type': 'lambertian', 'albedo': 0.2}

    with open('UVSPEC_AEROSOL_AUTO.INP', 'w') as f:
        f.write('atmosphere_file ../data/atmmod/afglus.dat\n')
        f.write(f'rte_solver {solver}        # Radiative transfer equation solver\n')
        if solver == 'polradtran':
            if polarized:
                f.write('polradtran nstokes 3\n')
            else:
                f.write('polradtran nstokes 1\n')
        if solver == 'mystic':
            f.write(f'mc_photons {mc_photons}\n')
            f.write('mc_vroom off \n')
        # Location of the extraterrestrial spectrum
        f.write('source solar ../data/solar_flux/atlas_plus_modtran\n')
        if brdf_dict['type'] == 'lambertian':
            f.write(f"albedo {brdf_dict['albedo']}\n")
        elif brdf_dict['type'] == 'rossli':
            f.write(f"brdf_ambrals iso {brdf_dict['iso']}\n")
            f.write(f"brdf_ambrals vol {brdf_dict['vol']}\n")
            f.write(f"brdf_ambrals geo {brdf_dict['geo']}\n")
        elif brdf_dict['type'] == 'rpv':
            f.write(f"brdf_rpv rho0  {brdf_dict['rho0']}\n")
            f.write(f"brdf_rpv k     {brdf_dict['k']}\n")
            f.write(f"brdf_rpv theta {brdf_dict['theta']}\n")
            f.write(f"brdf_rpv sigma {brdf_dict['sigma']}\n")
            f.write(f"brdf_rpv t1 {brdf_dict['t1']}\n")
            f.write(f"brdf_rpv t2 {brdf_dict['t2']}\n")
            f.write(f"brdf_rpv scale {brdf_dict['scale']}\n")

        f.write('sza ' + str(sza) + '\n')  # Specify gamma size distribution (alpha=7)
        f.write(f'phi0 {phi_0}                  # Solar azimuth angle\n')
        f.write('phi ' + array_to_str(phi) + '\n')  # Calculate all phase matrix elements
        f.write('umu ' + array_to_str(umu) + '\n')
        f.write('zout ' + str(zout) + ' \n')
        f.write('wavelength ' + str(wavelength) + '\n')  # Specify effective radius grid
        f.write(f"number_of_streams  {round(streams)}     # Number of streams\n")
        # f.write('spline 300 340 1         # Interpolate from first to last in step')
        # based on the UVSPEC_DISTORT.INP file which imports UVSPEC CLEAR
        # f.write('disort_intcor moments       # use Legendre coefficients for intensity corrections\n')
        f.write('wc_file 1D ../examples/WC.DAT\n')
        f.write(f'wc_modify tau set {cod}.\n')
        # f.write(f'wc_modify ssa set 0.9\n')
        f.write("print_disort_info 1 3 5\n")
        # f.write('output_user lambda eglo\n')
        f.write('quiet\n')
