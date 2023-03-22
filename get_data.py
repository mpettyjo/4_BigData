#! /usr/bin/env python3

'''
This program contains all the functions needed to parse data from Gaussian Output files
for the BigData1 paper.
'''

# import python packages for use in this program
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem

from mendeleev import element

#import chemcoord as cc
from scipy import stats
from rdkit import Chem
import seaborn as sns
import pandas as pd
import numpy as np
import shutil
import glob
import csv
import ast
import os
import re
from os.path import exists

#global variables
CONF, conf = 'Conformers', 'conformers'


def get_harm_info(file):
    ''' Get harmonic frequencies from Gaussian output file.
    
        Function that takes a file pathway, loads it, and extracts
        the harmonic frequencies and intensities from it.

        Parameters
        ----------
        file: string
              Pathway to a Gaussian vibrational frequency output file.

        Returns
        -------
        harm_info: list of lists containing floats
                    List of lists, one containing frequencies and the other 
                    the corresponding intensies
        '''
    inp = open(file, 'r')
    
    chk_count = 0
    for g in inp:
        if 'Normal termination' in g:
            chk_count = chk_count + 1
    inp.close()
    # Continues for output files that terminated normally (they must have 2 'Normal termination')
    if chk_count >= 2:
        # Obtains the raw vibrational data.
        inp = open(file, 'r')
        harm_info = []
        for line in inp:
            line = line.strip()
            if line.startswith('Frequencies --'):
                harm_info.append(line.split())
            elif line.startswith('IR Inten    --'):
                harm_info.append(line.split())
    # Prints molecule's name if there are less/more than 2 'Normal termination' in the output file.
    else:
        print(file.split('/')[-1]+' Normal termination problem.')
    inp.close()
    
    return harm_info

def unique_conf_list(location, loc_num_atoms):
    ''' Makes a list of all unique molecules and their conformers.
    
        Function that looks at all vibrational calculation output files and
        creates a list of tuples of form (string,List).

        Parameters
        ----------
        location: string
                   Directory pathway to the project folder, i.e., "4_BigData"
        loc_num_atoms: string
                       Directory name with form "{NUM}_Atoms/", where NUM is the
                       number of atoms.

        Returns
        -------
        unique_list: list of tuples
                     List of tuples of the form (string, List), where the first argument
                     is the unique molecule name and the second argument is a list of all 
                     conformers associated with that molecule.
        '''
    files_names_confs = glob.glob(
        location + loc_num_atoms +
        '4_VibrationalCalcs/OutputFiles/*.log')

    # Make lists for conformers
    filename_confs_list = [
        (file_vib.split('/')[-1].split('_harmonic')[0].split('_CONF')[0],
         file_vib.split('/')[-1].split('_harmonic')[0].split('_')[-1])
        for file_vib in files_names_confs
    ]

    filename_confs_list = list(zip(*filename_confs_list))

    unique_confs_filenames, unique_confs_counts = np.unique(list(
        filename_confs_list[0]),
                                                            return_counts=True)

    file_conf_comb_list = []
    for unique_mol, unique_count in zip(unique_confs_filenames,
                                        unique_confs_counts):
        conf_indices = np.where(
            np.asarray(filename_confs_list[0]) == unique_mol)[0]
        conf_list = []
        for ind in conf_indices:
            conf_num = np.asarray(filename_confs_list[1])[ind]
            conf_list.append(conf_num)
        if len(conf_list) == unique_count:
            file_conf_comb_list.append((unique_mol, conf_list))

    unique_list = file_conf_comb_list 

    return unique_list

def exp_to_sci(x):
    '''Convert very small exponential to scientific notation.

    From https://stackoverflow.com/questions/50679895/np-expx-when-x-is-small
    
    Parameters 
    ----------
    x: float
        Very small number.

    Returns
    --------
    (base_coeff,exponent): (float, float)
                            The coefficient and exponent of scientific notation
                            for a very small number.
    '''
    coeff, exp = np.modf(x / np.log(10.0))
    return 10**(coeff + 1), exp - 1

def get_boltzmannweights_at_T(final_df):
    ''' Calculate the Boltzmann Weights at a given set of temperatures.

        Function that computes the Boltzmann Weights from the Gibbs Free
        Energies at 4 temperatures in Kelvin, 10, 50, 100, 298.15, and populates
        `final_df` witht the calculated Boltzmann Weights.

        Parameters
        ----------
        final_df: pandas dataframe
                  Dataframe containing the Gibbs Free energies of all the conformers
                  of a given molecule.
        '''
    G_list = final_df['RawG'].values
    for T in [10,50,100,298.15]:
        coeff_list = []
        exp_list = []
        for G in G_list:
            b_exp=float(G/(3.166811563e-6*T))
            coeff,exp = exp_to_sci(-b_exp)
            coeff_list.append(coeff)
            exp_list.append(exp)
        P_list = []
        if exp_list.count(exp_list[0]) == len(exp_list):
            P_list = ["{:.2f}".format(coeff*100/sum(coeff_list)) for coeff in coeff_list]
        elif exp_list.count(exp_list[0]) != len(exp_list):
            n_list = np.max(exp_list)-exp_list
            n_coeff_list = list(zip(n_list,coeff_list))
            
            coeff_list_scaled = [n_coeff[1]/10**n_coeff[0] for n_coeff in n_coeff_list]
            
            P_list = ["{:.2f}".format(coeff*100/sum(coeff_list_scaled)) for coeff in coeff_list_scaled]
        
        col_name= str('BW Percent at '+str(T)+' K')
        final_df[col_name] = P_list
    return

## These two functions work together ##
def int_get_gibbs_energy(location,loc_num_atoms,mol_id,CONF,conf,energy_df):
    ''' Extract the Gibbs free energies from the CENSO output files for 
        one molecule and then put then in a dataframe.

        Parameters
        ----------
        location: string
                    Directory pathway to the project folder, i.e., "4_BigData"
        loc_num_atoms: string
                        Directory name with form "{NUM}_Atoms/", where NUM is the
                        number of atoms.
        mol_id: string
                Molecule id as given in biosignature data set file.
        CONF: string
                Global variable defined above.
        conf: string
                Global variable defined above.
        energy_df: pandas dataframe
                    Empty dataframe containing the column names 'CONF#','G','RawG'

        Returns
        -------
        energy_df: pandas dataframe
                    Dataframe containing the Gibbs free energies of all of the conformers
                    for `mol_id.`
    '''
    file = location+loc_num_atoms+'3_CREST_CENSO_Outputs/BWeights_GeomFiles_'+CONF+'/'+mol_id+'_bweight_'+conf+'.dat'
    print(file)
    if os.path.exists(file) == True:
        inp = open(file, 'r')
        
        energy_info = []
        for line in inp:
            line = line.strip()
            if line.startswith('CONF'):
                energy_info.append(line.split())
        for row in energy_info:
            if energy_info.index(row) > 0:
                if 'Conf' in CONF: 
                    energy_df.loc[len(energy_df)] = [row[0],float(row[5])*627.509*4.184,float(row[5])] # 4.184 is the conversion factor to kJ/mol
    return energy_df

def get_censo_gibbs_energy(location,loc_num_atoms,mol_id):
    ''' Create dataframe with Gibbs free energy and Boltzmann weights
        for a given molecule.
    
        Implements int_get_gibbs_energy above to create a dataframe
        containing the conformer ids, Gibbs free energies, and
        Boltzmann Weights for the molecule given by `mol_id.`

    Parameters
    ----------
    location: string
                Directory pathway to the project folder, i.e., "4_BigData"
    loc_num_atoms: string
                    Directory name with form "{NUM}_Atoms/", where NUM is the
                    number of atoms.
    mol_id: string
            Molecule id as given in biosignature data set file.

    Returns
    -------
    '''
    energy_df = pd.DataFrame(columns = ['CONF#','G','RawG'])
    CONF = 'Conformers'
    conf = 'conformers'
    
    final_df = int_get_gibbs_energy(location,loc_num_atoms,mol_id,CONF,conf,energy_df)
        
    
    final_df = final_df.sort_values('G')
    min_energy = np.min(final_df['G'].values)
    final_df['G'] = final_df['G'] - min_energy

    get_boltzmannweights_at_T(final_df)
    
    return final_df
## end of two functions ##

def get_dipole_moments(file):
    ''' Get dipole moments from Gaussian output file.

    Parameters
    ----------
    file: string
              Pathway to a Gaussian vibrational frequency output file.

    Returns
    -------
    [abc_dipole, total_dipole, a_dipole, b_dipole, c_dipole]: [float, float, float, float, float]
                List of dipole moments of molecule in Debye.
    '''
    dipole_info = []
    rotmatrix_info = []
    mat_count = 0
    with open(file, 'r') as inp:
        for line in inp:
            line = line.strip()

            if line.startswith('Dipole moment (Debye):'):
                #print(next(inp).split())
                dipole_info.append(next(inp).split())
            if line.startswith('Rotation matrix to Principal Axis frame:'):
                mat_count+=1
                if mat_count==2:
                    rotmatrix_info.append(next(inp).split())
                    rotmatrix_info.append(next(inp).split())
                    rotmatrix_info.append(next(inp).split())
                    rotmatrix_info.append(next(inp).split())
        
    row = dipole_info[-1]
    xyz_dipole = [float(row[0]),float(row[1]),float(row[2])]
    rotmatrix_info.remove(['1', '2', '3'])

    rot_matrix = []
    for row in rotmatrix_info:
        rot_matrix.append([float(row[1].replace('D','E')),float(row[2].replace('D','E')),float(row[3].replace('D','E'))])
    abc_dipole = np.matmul(rot_matrix,xyz_dipole)
    
    return np.append(abc_dipole,[np.sqrt(abc_dipole[0]**2+abc_dipole[1]**2+abc_dipole[2]**2),abc_dipole[0]*abc_dipole[1]*abc_dipole[2]])


## Functions for getting rotational data ##
def get_rotational_constants(file):
    ''' Get A, B, C constants from Gaussian output file.
    
    Parameters
    ----------
    file: string
              Pathway to a Gaussian vibrational frequency output file.
    
    Returns
    -------
    [A, B, C]: [float, float, float]
                List containing A, B, and C rotational constants in cm-1.
    '''
    rot_count = 0
    with open(file, 'r') as inp:
        for line in inp:
            line = line.strip()
            if line.startswith('Rotational constants (MHZ):'):
                rot_count+=1
                if rot_count==2:
                    rot_info=next(inp).split()
    if (rot_info[0] == '***************'):
        rot_info[0] = 0
    elif (rot_info[1] == '***************'):
        rot_info[1] = 0
    elif (rot_info[2] == '***************'):
        rot_info[2] = 0
        
    return [float(rot_info[0])/29979.2458,float(rot_info[1])/29979.2458,float(rot_info[2])/29979.2458]

def grab_constant(inp,quartic_constants):
    '''Get quartic rotational constant from list and add
        them to the input list.
    
        Parameters
        ----------
        inp: opened file
                Gaussian output file.
        quartic_constants: List
                            Empty list to be populated by quaritic constants.

        '''
    rot_info = next(inp).split()
    quartic_constants.append(float(rot_info[3])/29979.2458)
    return

def get_quartic_rotational_constants(file):  #NOT DONE
    ''' Get quartic constants from a Gaussian input file.

    Parameters
    ----------
    file: string
              Pathway to a Gaussian vibrational frequency output file.

    Returns
    -------
    [Delta_N, ]: [float, float, float, float, float]
                List containing Delta_N, ... quartic rotational constants 
                in cm-1.
    '''
    quartic_constants = []
    with open(file, 'r') as inp:
        for line in inp:
            line = line.strip()
            if line.startswith('(Asymmetrically reduced)          (Symmetrically reduced)'):
                for n in range(0,5,1):
                    grab_constant(inp,quartic_constants)
    print(quartic_constants)
    return 

def have_hyperfine(file):
    ''' Check if molecule has hyperfine data in Gaussian vibrational 
        calculation output file.

    Parameters
    ----------
    file: string
              Pathway to a Gaussian vibrational frequency output file.

    Returns
    -------
    hyperfine: string
                String "Yes" or "No" indicating if hyperfine data exists.
    '''
    with open(file, 'r') as inp:
        for line in inp:
            line = line.strip()
            if line.startswith('Atoms with significant hyperfine tensors:'):
                print(line.split())
                if len(line.split()) == 5:
                    hyperfine = 'No'
                    print(hyperfine)
                elif len(line.split()) > 5:
                    hyperfine = 'Yes'
                    print(hyperfine)
    return hyperfine

## end of rotational data functions ##

def get_rotamer_degeneracy(location,loc_num_atoms,mol_id,conf):
    ''' Get rotamer degeneracy from CREST output file for molecule.

    Parameters
    ----------
    location: string
                Directory pathway to the project folder, i.e., "4_BigData"
    loc_num_atoms: string
                    Directory name with form "{NUM}_Atoms/", where NUM is the
                    number of atoms.
    mol_id: string
            Molecule id as given in biosignature data set file.
    conf: string
            Global variable defined above.

    Returns
    -------
    rota_degen: int
                Rotamer degeneracy of a molecule.
    '''
    file_crest = location+loc_num_atoms+'3_CREST_CENSO_Outputs/CREST_OutputFiles/'+mol_id+'_crest.out'
    degen_info_list = []
    with open(file_crest, 'r') as inp:
        for line in inp:
            line = line.strip()
            if line.startswith('Erel/kcal'):
                while not all(i in ['T', '/K', ':', '298.15'] for i in line):
                    line=next(inp).split()
                    degen_info_list.append(line)     
    inp.close()
    
    conf_num = conf.split('CONF')[-1]
    rota_degen=1
    for line in degen_info_list:
        if len(line)==8:
            if conf_num==line[-3]:
                rota_degen = line[-2]
    print(mol_id,conf_num,rota_degen)            
    return int(rota_degen)

# -------------------------------------------
# Parsing Vibrational Data from Output Files
# -------------------------------------------

## these two functions work together ##
def int_parse_vib_data(location,loc_num_atoms,biosig_data,df):
    ''' Get all data from Gaussian output files for a given number of atoms
        folder and store in a dataframe. 

        Parameters
        ----------
        location: string
                        Directory pathway to the project folder, i.e., "4_BigData"
        loc_num_atoms: string
                        Directory name with form "{NUM}_Atoms/", where NUM is the
                        number of atoms.
        biosig_data: pandas dataframe
                    Dataframe containing all molecules in Seager et al.'s data set.
        df: pandas dataframe
            Dataframe with columns according to data to be added.
        
        Returns
        -------
        df: pandas dataframe
            Dataframe containing data extracted from Gaussian vibrational frequency 
            calculation output files.
    '''
    file_conf_list=unique_conf_list(location,loc_num_atoms)

    # Opens each individual output file as reference for the SMILES codes.
    for mol in file_conf_list:
        mol_id = mol[0]
        mol_conf = mol[1]
        

        # Gets the SMILES code and the experimental frequencies from the benchmark dataset.
        smiles_code = biosig_data.query("Formula_ID == @mol_id")['SMILES'].iloc[0] 
        mol_tot_atoms = int(biosig_data.loc[biosig_data['Formula_ID'] == mol_id]['Tot_atoms'].values[0])

        # Gets dataframe with gibbs energy for each Conf
        energy_df = get_censo_gibbs_energy(location,loc_num_atoms,mol_id)

        inter_row_info_list = []
        for conf in mol_conf:
            
            # get Gibbs energies
            conf_energy = energy_df.loc[energy_df['CONF#'] == conf]['G'].values[0]
            conf_bw_10K = energy_df.loc[energy_df['CONF#'] == conf]['BW Percent at 10 K'].values[0]
            conf_bw_50K = energy_df.loc[energy_df['CONF#'] == conf]['BW Percent at 50 K'].values[0]
            conf_bw_100K = energy_df.loc[energy_df['CONF#'] == conf]['BW Percent at 100 K'].values[0]
            conf_bw_29815K = energy_df.loc[energy_df['CONF#'] == conf]['BW Percent at 298.15 K'].values[0]

            conf_type = re.split('(\d+)',conf)[0]

            if conf_type == 'CONF':
                file = location+loc_num_atoms+'4_VibrationalCalcs/OutputFiles/'+mol_id+'_'+conf+'_harmonic_conformers.log'

            # get harmonic frequencies from 'file' for specific Conf
            harm_info = get_harm_info(file)


            conf_raw_freqs = []
            conf_scaled_freqs = []
            conf_ints = []

            for line_data in harm_info:
                # From the raw vibrational data obatins the frequencies and modes.
                if line_data[0] == 'Frequencies':
                    for index,value in enumerate(line_data):
                        if index > 1:
                            #counter += 1
                            conf_raw_freqs.append(float(value))
                            #hmodes.append(str(counter))

                # From the raw vibrational data obtained the intensities.
                elif line_data[0] == 'IR':
                    for index,value in enumerate(line_data):
                        if index > 2:
                            conf_ints.append(float(value))


            # test for imaginary frequencies.
            neg_freqs = 0
            for freq in conf_raw_freqs:
                if float(freq) < 0:
                    neg_freqs = neg_freqs + 1

            scale_factors = [0.995395864,0.979158827,0.963816253] #[low,mid,high] #B971/Def2TZVPD
            # If no imaginary frequencies, prints data into a dataframe (later csv file).
            if neg_freqs == 0:
                conf_scaled_freqs = []
                for freq in conf_raw_freqs:
                    if freq < 1000:
                        freq_scaled = freq*scale_factors[0]
                    elif (freq >= 1000) and (freq <= 2000):
                        freq_scaled = freq*scale_factors[1]
                    elif freq > 2000:
                        freq_scaled = freq*scale_factors[2]
                    conf_scaled_freqs.append(freq_scaled)

                # Get dipole moments
                dipole_info = get_dipole_moments(file)
                
                # Get Rotational Constants in cm-1
                #print(file)
                rot_info = get_rotational_constants(file)
                
                # Get rotamer degeneracy
                rota_degen = get_rotamer_degeneracy(location,loc_num_atoms,mol_id,conf)

                # Make each row of data
                row_info = list([mol_id,smiles_code,mol_tot_atoms,conf,rota_degen,conf_energy,conf_bw_10K,conf_bw_50K,conf_bw_100K,conf_bw_29815K,conf_raw_freqs,conf_scaled_freqs,conf_ints,dipole_info[0],dipole_info[1],dipole_info[2],dipole_info[3],dipole_info[4],rot_info[0],rot_info[1],rot_info[2]])
                df.loc[len(df)] = row_info

            # If imaginary frequency found, prints the name of the molecule.
            
            else:
                print(file.split('/')[-1]+' has imaginary frequencies.')
    return df
    
def parse_vib_data(location,biosig_data,df):
    ''' Gets data from Gaussian output files for all atoms folders using
        in_parse_vib_data and populates pandas dataframe `df`.

        Parameters
        ----------
        location: string
                        Directory pathway to the project folder, i.e., "4_BigData"
        loc_num_atoms: string
                        Directory name with form "{NUM}_Atoms/", where NUM is the
                        number of atoms.
        biosig_data: pandas dataframe
                    Dataframe containing all molecules in Seager et al.'s data set.
        df: pandas dataframe
            Empty dataframe with columns according to data to be added.
    '''
    files_list = glob.glob(comp_loc+'4_BigData/?_Atoms')+glob.glob(comp_loc+'4_BigData/??_Atoms')
    files_list.remove(comp_loc+'4_BigData/21_Atoms')# Not Complete yet
    print(files_list)
    for file in files_list:
        loc_num_atoms = file.split('/')[-1]+'/'
        int_parse_vib_data(location,loc_num_atoms,biosig_data,df)
    return
## end of two functions ##