from typing import Callable
import numpy as np
import torch as th

"""
class scalarize_objectives:
    def __init__(self,
                sf : Callable):
        self._sf = sf
        #self.weight_vector = weight_vector
        #self.z_ideal = z_ideal
        #self.z_nadir = z_nadir

    def evaluate(self,
                fitness, 
                weight_vector : np.ndarray, 
                z_ideal : np.ndarray = None, 
                z_nadir : np.ndarray = None):
        self._sf(fitness, weight_vector, z_ideal, z_nadir)
    
    def scalarize_WS(self, fitness: np.ndarray):
        weights = weights/weights.sum()
        if th.is_tensor(fitness):
            tens_weights = th.Tensor(weights).cuda()
            fitness_product = th.mul(fitness,tens_weights)
            fitness_sum = fitness_product.sum(1)
        else:
            fitness_product = np.multiply(fitness,weights)
            fitness_sum = fitness_product.sum(axis=1)
        return fitness_sum
    
    def scalarize_ASF(fitness_vector: np.ndarray, 
                    weights: np.ndarray, 
                    nadir_point: np.ndarray,
                    utopian_point: np.ndarray,
                    rho: float = 1e-6):
        #### The weight vectors should be normalized and adapted based on the objective values
        f = fitness_vector
        rho = rho
        z_nad = nadir_point
        z_uto = utopian_point
        #mu = weights / (z_nad - z_uto)
        #max_term = np.max(mu * (f - z_uto), axis=-1)
        #sum_term = rho * np.sum((f - z_uto) / (z_nad - z_uto), axis=-1)
        mu = weights
        max_term = np.max(mu * (f - z_uto), axis=-1)
        sum_term = rho * np.sum((f - z_uto), axis=-1)

        return max_term + sum_term

"""

def scalarize(sf, fitness, weights, z_ideal=None, z_nadir=None):
    if sf == "WS":
        return scalarize_WS(fitness, weights)
    elif sf=="ASF":
        return scalarize_ASF(fitness, weights,z_nadir, z_ideal)
    else:
        return scalarize_TCH(fitness, weights, z_ideal)

def scalarize_index(sf, fitness, weights, z_ideal=None, z_nadir=None):
    if sf == "WS":
        return scalarize_WS(fitness, weights)
    elif sf=="ASF":
        return scalarize_ASF(fitness, weights,z_nadir, z_ideal)
    else:
        return scalarize_TCH_index_step(fitness, weights, z_ideal, z_nadir)


def scalarize_WS(fitness, weights):
    weights = weights/weights.sum()
    if th.is_tensor(fitness):
        tens_weights = th.Tensor(weights).cuda()
        fitness_product = th.mul(fitness,tens_weights)
        fitness_sum = fitness_product.sum(1)
    else:
        fitness_product = np.multiply(fitness,weights)
        fitness_sum = fitness_product.sum(axis=1)
    return fitness_sum

def get_pbi_weights(solution, ideal_point, weights, theta):
    w = np.array(weights)
    z_star = np.array(ideal_point)
    F = np.array(solution.objectives)

    d1_vect = np.dot((F - z_star), w)
    d1 = np.linalg.norm(d1_vect) / np.linalg.norm(w)
    d2_vect = F - (z_star + d1 * w)
    d2 = np.linalg.norm(d2_vect)
    return (d1 + theta * d2).tolist()


"""
def scalarize_HV_tensor(fitness, ref_point, num_envs):
    ref_weights = th.Tensor(ref_point).cuda()
    #fitness_product = th.mul(fitness,tens_weights)
    fitness_product = fitness*tens_weights.tile((num_envs,1)).transpose(0,1)
    fitness_sum = fitness_product.sum(1)
    return fitness_sum
"""

def scalarize_WS_tensor(fitness, weights, num_envs, device):
    weights = weights/weights.sum()
    tens_weights = th.Tensor(weights).to(device)
    #fitness_product = th.mul(fitness,tens_weights)
    fitness_product = fitness*tens_weights.tile((num_envs,1)).transpose(0,1)
    fitness_sum = fitness_product.sum(1)
    return fitness_sum


def scalarize_ASF(fitness_vector: np.ndarray, 
                weights: np.ndarray, 
                nadir_point: np.ndarray,
                utopian_point: np.ndarray,
                rho: float = 1e-6):
    #### The weight vectors should be normalized and adapted based on the objective values
    f = fitness_vector
    rho = rho
    z_nad = nadir_point
    z_uto = utopian_point
    scale = z_nad - z_uto
    mu = weights
    if th.is_tensor(fitness_vector):
        pass
        #tens_weights = th.Tensor(weights).cuda()

    else:    
        #mu = weights / (z_nad - z_uto)
        #max_term = np.max(mu * (f - z_uto), axis=-1)
        #sum_term = rho * np.sum((f - z_uto) / (z_nad - z_uto), axis=-1)
        max_term = np.max(mu * (f - z_uto)/scale, axis=-1)
        sum_term = rho * np.sum((f - z_uto)/scale, axis=-1)

    return max_term + sum_term

def scalarize_TCH(fitness: np.ndarray, 
                weights: np.ndarray,
                z_ideal: np.ndarray):
    if th.is_tensor(fitness):
        tens_weights = th.Tensor(weights).cuda()
        z_ideal_tens = th.Tensor(z_ideal).cuda()
        fitness_sub = th.sub(-fitness, z_ideal_tens)
        fitness_norm = th.linalg.norm(fitness_sub,dim=1)
        fitness_norm2 = fitness_norm.repeat(fitness.size()[1],1).transpose(0,1)
        #scalar_fitness = th.max(th.mul(fitness_sub/fitness_norm2,tens_weights),1)[0]
        scalar_fitness = th.max(th.mul(fitness_sub,tens_weights),1)[0]
        scalar_fitness = -scalar_fitness
    else:
        fitness_norm = np.linalg.norm(-fitness - z_ideal, axis=1)
        fitness_norm2 = np.repeat(fitness_norm, np.shape(fitness)[1]).reshape(np.shape(fitness))
        #scalar_fitness = -np.max((weights * (-fitness- z_ideal)/fitness_norm2), axis=1)
        scalar_fitness = -np.max((weights * (-fitness- z_ideal)), axis=1)
    return scalar_fitness

def scalarize_TCH_index(fitness: np.ndarray, 
                weights: np.ndarray,
                z_ideal: np.ndarray,
                z_nadir: np.ndarray):
    #print(fitness)
    weights = weights/weights.sum()
    if not th.is_tensor(fitness):
        fitness = th.Tensor(fitness).cuda()
    fitness = fitness.sum(0)
    tens_weights = th.Tensor(weights).cuda()
    z_ideal_tens = th.Tensor(z_ideal).cuda()
    norm_factor = th.Tensor((-z_nadir+z_ideal)).cuda()
    fitness_sub = th.sub(-fitness, z_ideal_tens)
    #fitness_norm = th.linalg.norm(fitness_sub,dim=1)
    #fitness_norm2 = fitness_norm.repeat(fitness.size()[1],1).transpose(0,1)
    #scalar_fitness = th.max(th.mul(fitness_sub/fitness_norm2,tens_weights),1)[0]
    fitness_norm = th.div(fitness_sub, norm_factor)
    scalar_fitness_index = th.argmax(th.mul(fitness_norm,tens_weights))
    #scalar_fitness = -scalar_fitness
    #return scalar_fitness_index
    
    #else:
    #    fitness_norm = np.linalg.norm(-fitness - z_ideal, axis=1)
    #    fitness_norm2 = np.repeat(fitness_norm, np.shape(fitness)[1]).reshape(np.shape(fitness))
        #scalar_fitness = -np.max((weights * (-fitness- z_ideal)/fitness_norm2), axis=1)
    #    scalar_fitness = -np.max((weights * (-fitness- z_ideal)), axis=1)
    return scalar_fitness_index

def scalarize_TCH_index2(fitness: np.ndarray, 
                weights: np.ndarray,
                z_ideal: np.ndarray,
                z_nadir: np.ndarray):
    #print(fitness)
    if not th.is_tensor(fitness):
        fitness = th.Tensor(fitness).cuda()

    #fitness = fitness.sum(0)
    fitness = fitness.min(axis=0)[0]
    tens_weights = th.Tensor(weights).cuda()
    z_ideal_tens = th.Tensor(z_ideal).cuda()
    z_uto = th.Tensor(np.array([0,0])).cuda()
    norm_factor = th.Tensor((-z_nadir+z_ideal)).cuda()
    fitness_norm = th.div(th.sub(-fitness, z_ideal_tens), norm_factor)
    fitness_sf = th.mul(th.sub(fitness_norm, z_uto), tens_weights)
    scalar_fitness_index = th.argmax(fitness_sf)
    return scalar_fitness_index

def scalarize_TCH_index3(fitness: np.ndarray, 
                weights: np.ndarray,
                z_ideal: np.ndarray,
                z_nadir: np.ndarray):
    #print(fitness)
    if not th.is_tensor(fitness):
        fitness = th.Tensor(fitness).cuda()

    #fitness = fitness.sum(0)
    fitness = fitness.min(axis=0)[0]
    tens_weights = th.Tensor(weights).cuda()
    z_ideal_tens = th.Tensor(z_ideal).cuda()
    norm_factor = th.Tensor((-z_nadir+z_ideal)).cuda()
    #fitness_sub = th.sub(-fitness, z_ideal_tens)
    fitness_sub = th.sub(-fitness, tens_weights)
    fitness_norm = th.div(fitness_sub, norm_factor)
    #scalar_fitness_index = th.argmax(th.mul(fitness_norm,tens_weights))
    scalar_fitness_index = th.argmax(fitness_norm)
    
    #scalar_fitness = -scalar_fitness
    #return scalar_fitness_index
    
    #else:
    #    fitness_norm = np.linalg.norm(-fitness - z_ideal, axis=1)
    #    fitness_norm2 = np.repeat(fitness_norm, np.shape(fitness)[1]).reshape(np.shape(fitness))
        #scalar_fitness = -np.max((weights * (-fitness- z_ideal)/fitness_norm2), axis=1)
    #    scalar_fitness = -np.max((weights * (-fitness- z_ideal)), axis=1)
    return scalar_fitness_index

def scalarize_TCH_index_step(fitness: np.ndarray, 
                weights: np.ndarray,
                z_ideal: np.ndarray,
                z_nadir: np.ndarray):

    if not th.is_tensor(fitness):
        fitness = th.Tensor(fitness).cuda()
    tens_weights = th.Tensor(weights).cuda()
    z_ideal_tens = th.Tensor(z_ideal).cuda()
    norm_factor = th.Tensor((-z_nadir+z_ideal)).cuda()
    fitness_sub = th.sub(-fitness, z_ideal_tens)
    fitness_norm = th.div(fitness_sub, norm_factor)
    scalar_fitness_index = th.argmax(th.mul(fitness_norm,tens_weights), axis=1).cpu().numpy()

    return scalar_fitness_index