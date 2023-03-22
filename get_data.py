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
        CONF: string
        conf: string
        energy_df: pandas dataframe

        Returns
        -------
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
    '''
    Parameters
    ----------

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