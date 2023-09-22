import numpy as np
import matplotlib.pyplot as plt

class SJAYA:
    def __init__(self, NumOfDesigns, SizeofPopulation): 
        self.NumOfDesigns = NumOfDesigns
        self.SizeofPopulation = SizeofPopulation

    def PerformSJAYA(self, NumOfParameters, ParaRange, save_path=None):
        lower_limits = ParaRange[0]
        upper_limits = ParaRange[1]
        population = np.zeros((self.NumOfDesigns, self.SizeofPopulation,NumOfParameters))
        for i in range(self.NumOfDesigns):
            SubPopulations = np.zeros((self.SizeofPopulation,NumOfParameters))
            for j in range(NumOfParameters):
                SubPopulations[:,j] = np.random.uniform(lower_limits[j], upper_limits[j],size=self.SizeofPopulation)
            population[i,:,:] = SubPopulations
        cost = self.costFunction(ParaRange, population[0], self.NumOfDesigns)
        allBestDesign = self.initialSamples(ParaRange, population, self.NumOfDesigns, "min")
        allWorstDesign = self.initialSamples(ParaRange, population, self.NumOfDesigns, "min")

        for iterations in range(50):
            for popIndx in range(population.shape[0]):
                population, allBestDesign, allWorstDesign = self.JayaAlgorithm(population, allBestDesign, allWorstDesign, popIndx, ParaRange, self.NumOfDesigns)
        plt.figure()
        plt.scatter(allBestDesign[:,0], allBestDesign[:,1])
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def scaling(self, paraRange, arr):
        lower_limits = paraRange[0]
        upper_limits = paraRange[1]
        arr_scaled = arr
        for i in range(len(paraRange[1])):
            arr_scaled[:, i] = (arr[:, i] - lower_limits[i]) / (upper_limits[i] - lower_limits[i])
        return arr_scaled

    def costFunction(self, paraRange, arr, numOfDes):
        arr = self.scaling(paraRange, arr)
        sf = 0
        for i in range(arr.shape[0]):
            sf = sf + self.inverse_euclidean_distance(arr[i,:], arr)
        nc = self.noncollapsing(paraRange, arr, numOfDes)
        return (sf + nc)

    def inverse_euclidean_distance(self, array1D, array2D):
        distances = np.sum((array1D - array2D) ** 2, axis=1)

        # Handle division by zero
        epsilon = np.finfo(float).eps
        safe_distances = np.where(distances == 0, epsilon, distances)
        inverse_distances = 1 / safe_distances
        inverse_distances[distances == 0] = 0

        sum_inverse_distances = np.sum(inverse_distances)
        return sum_inverse_distances

    def noncollapsing(self,paraRange, arr, numOfDes):
        nc = 0
        rd = np.linspace(0, 1, numOfDes + 1)
        rd = rd[:-1] + (rd[1] / 2)
        lower_limits = np.array(paraRange[0])
        upper_limits = np.array(paraRange[1])
        intervals = lower_limits + np.expand_dims(rd, axis=-1) * (upper_limits - lower_limits)
        discreate = np.zeros((arr.shape[0], arr.shape[1]))
        for i in range(arr.shape[0]):
            discreate[i, :] = np.argmin(np.abs(arr[i, :] - intervals), axis=0)
        discreate = discreate - 1
        for i in range(arr.shape[0]):
            ts = np.zeros((arr.shape[0] - i, arr.shape[1]))
            ts[np.where(np.abs(discreate[i, :] - discreate[i + 1:, :]) == 0)] = 1
            nc = nc + np.sum(ts)
        return nc

    def initialSamples(self, paraRange, arr3D, numOfDes, type):
        D = np.repeat(np.arange(0,arr3D.shape[1]), arr3D.shape[1])
        B = np.tile(np.arange(0,arr3D.shape[1]), arr3D.shape[1])
        C = np.array([D,B])
        sf = np.zeros(B.shape[0])
        for i in range(D.shape[0]):
            sf[i] = self.costFunction(paraRange, np.array([arr3D[0,C[0,i],:],arr3D[1,C[1,i],:]]), numOfDes)
        if (type == "min"):
            ind = np.unravel_index(np.argmin(sf, axis=None), sf.shape)
        elif (type == "max"):
            ind = np.unravel_index(np.argmax(sf, axis=None), sf.shape)
        ini_samples = np.append(arr3D[0,C[0,ind],:], arr3D[1,C[1,ind],:], axis=0)
        for i in range(2,arr3D.shape[0]):
            sf = np.zeros(B.shape[0])
            for j in range(arr3D.shape[1]):
                new_sample = arr3D[j,i,:].reshape(1, -1)
                sf[j] = self.costFunction(paraRange, np.append(ini_samples, new_sample, axis=0), numOfDes)
            if (type == "min"):
                ind = np.unravel_index(np.argmin(sf, axis=None), sf.shape)
            elif (type == "max"):
                ind = np.unravel_index(np.argmax(sf, axis=None), sf.shape)
            new_sample = arr3D[j,i,:].reshape(1, -1)
            ini_samples = np.append(ini_samples, new_sample, axis=0)
        return ini_samples

    def JayaAlgorithm(self, allPop, allBestDesign, allWorstDesign, popIndx, paraRange, noOfDesigns):
        lower_limits = np.array(paraRange[0])
        upper_limits = np.array(paraRange[1])
        pop = allPop[popIndx,:,:]
        bestDesign = allBestDesign[popIndx,:]
        worstDesign = allWorstDesign[popIndx,:]

        trainedPop = pop.copy()
        rand_1 = np.random.uniform(0, 1, pop.shape[1])
        rand_2 = np.random.uniform(0, 1, pop.shape[1])

        for i in range(pop.shape[0]):
            for j in range(pop.shape[1]):
                trainedPop[i, j] = pop[i, j] + (rand_1[j] * (bestDesign[j] - np.abs(pop[i, j]))) - (rand_2[j] * (worstDesign[j] - np.abs(pop[i, j])))
                
                if (trainedPop[i, j] < lower_limits[j] or trainedPop[i, j] > upper_limits[j]):
                    trainedPop[i, j] = pop[i, j]

        temCost = np.zeros(pop.shape[0])
        for i in range(trainedPop.shape[0]):
            allBestDesign[popIndx, :] = pop[i, :]
            oldCost = self.costFunction(paraRange, allBestDesign, noOfDesigns)
            allBestDesign[popIndx, :] = trainedPop[i, :]
            newCost = self.costFunction(paraRange, allBestDesign, noOfDesigns)
            if (oldCost <= newCost):
                trainedPop[i, :] = pop[i, :]
                temCost[i] = oldCost
            else:
                temCost[i] = newCost
        
        allPop[popIndx, :, :] = trainedPop
        ind = np.argmin(temCost)
        allBestDesign[popIndx, :] = trainedPop[ind, :]

        ind = np.argmax(temCost)
        allWorstDesign[popIndx, :] = trainedPop[ind, :]

        return allPop, allBestDesign, allWorstDesign

