from typing import Callable, List
from warnings import warn

import numpy as np

def APD_performance(fitness, ref_vectors, penalty, n_of_fitnesses, refx, alpha: float = 2):
    if penalty < 0:
        penalty = 0
    if penalty > 1:
        penalty = 1
    partial_penalty_factor = (penalty ** alpha) * n_of_fitnesses
      
    ideal = np.amin(fitness, axis=0)
    #print("ideal:",self.ideal)
    #print("RVs:",self.vectors.values)
    translated_fitness = fitness - ideal
    fitness_norm = np.linalg.norm(translated_fitness, axis=1)
    # TODO check if you need the next line
    # TODO changing the order of the following few operations might be efficient
    fitness_norm = np.repeat(fitness_norm, len(translated_fitness[0, :])).reshape(
        len(fitness), len(fitness[0, :])
    )
    # Convert zeros to eps to avoid divide by zero.
    # Has to be checked!
    fitness_norm[fitness_norm == 0] = np.finfo(float).eps
    normalized_fitness = np.divide(
        translated_fitness, fitness_norm
    )  # Checked, works.
    cosine = np.dot(normalized_fitness, np.transpose(ref_vectors))
    if cosine[np.where(cosine > 1)].size:
        warn("RVEA.py line 60 cosine larger than 1 decreased to 1")
        cosine[np.where(cosine > 1)] = 1
    if cosine[np.where(cosine < 0)].size:
        warn("RVEA.py line 64 cosine smaller than 0 increased to 0")
        cosine[np.where(cosine < 0)] = 0
    # Calculation of angles between reference vectors and solutions
    theta = np.arccos(cosine)
    # Reference vector assignment
    assigned_vectors = np.argmax(cosine, axis=1)
    selection = np.array([], dtype=int)
    # Selection
    # Convert zeros to eps to avoid divide by zero.
    # Has to be checked!
    refx[refx == 0] = np.finfo(float).eps
    for i in range(0, len(ref_vectors)):
        sub_population_index = np.atleast_1d(
            np.squeeze(np.where(assigned_vectors == i))
        )

        sub_population_fitness = translated_fitness[sub_population_index]
        #print("subpop:",sub_population_fitness)
        # fast tracking singly selected individuals
        if len(sub_population_index) == 1:
            selx = sub_population_index
            if selection.shape[0] == 0:
                selection = np.hstack((selection, np.transpose(selx[0])))
            else:
                selection = np.vstack((selection, np.transpose(selx[0])))
        elif len(sub_population_index) > 1:
            # APD Calculation
            angles = theta[sub_population_index, i]
            angles = np.divide(angles, refx[i])  # This is correct.
            # You have done this calculation before. Check with fitness_norm
            # Remove this horrible line
            sub_pop_fitness_magnitude = np.sqrt(
                np.sum(np.power(sub_population_fitness, 2), axis=1)
            )
            apd = np.multiply(
                np.transpose(sub_pop_fitness_magnitude),
                (1 + np.dot(partial_penalty_factor, angles)),
            )
            minidx = np.where(apd == np.nanmax(apd))
            if np.isnan(apd).all():
                continue
            selx = sub_population_index[minidx]
            if selection.shape[0] == 0:
                selection = np.hstack((selection, np.transpose(selx[0])))
            else:
                selection = np.vstack((selection, np.transpose(selx[0])))
    
    #print("selection:",selection)
    return selection.squeeze()

   