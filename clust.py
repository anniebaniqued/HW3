# clust.py
# -------
# YOUR NAME HERE

import sys
import random
import numpy as np
import math
import utils
from operator import add
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

DATAFILE = "adults.txt"

#validateInput()

def validateInput():
    if len(sys.argv) != 3:
        return False
    if sys.argv[1] <= 0:
        return False
    if sys.argv[2] <= 0:
        return False
    return True


#-----------

# modified to also take numExamples
def parseInput(datafile, numExamples):
    """
    params datafile: a file object, as obtained from function `open`
    returns: a list of lists

    example (typical use):
    fin = open('myfile.txt')
    data = parseInput(fin)
    fin.close()
    """
    data = []
    for line in datafile:
        instance = line.split(",")
        instance = instance[:-1]
        data.append(map(lambda x:float(x),instance))
        if len(data) == numExamples:
            break
    return data


def printOutput(data, numExamples):
    for instance in data[:numExamples]:
        print ','.join([str(x) for x in instance])

def kMeans(data, numClusters):
    muList = []
    r_vectors = []

    for i in range(numClusters):
        muList.append(data[random.randint(0, len(data))])

    for i in range(len(data)):
        toAppend = []
        for k in range(numClusters):
            toAppend.append(0)
        r_vectors.append(toAppend)

    # convergence when no examples are reassigned
    somethingChanged = True
    while somethingChanged:
        # setting the r vector: assigning examples to clusters
        somethingChanged = False
        # average mean squared distance
        ave_msd = 0
        for i in range(len(data)):
            distances = []
            for muVec in muList:
                if not(isinstance(muVec,str)):
                    distances.append(utils.squareDistance(data[i], muVec))
            minDist = min(distances)
            ave_msd += minDist
            k = distances.index(minDist)
            rVec = r_vectors[i]
            if rVec[k] != 1: # data as been reassigned to different cluster 
                somethingChanged = True
            for j in range(len(rVec)):
                if j == k:
                    rVec[j] = 1
                else:
                    rVec[j] = 0
            r_vectors[i] = rVec
        ave_msd = ave_msd/len(data)

        # recenter the prototype vectors (muVecs)
        for k in range(len(muList)):
            count = 0.0
            vecSum = [0] * len(data[0]) # initialize vecSum to zero vector with the same length as a datum
            for i in range(len(data)):
                if r_vectors[i][k] == 1:
                    vecSum = map(add, vecSum, data[i])
                    count += 1.0
            if count!=0:
                muList[k] = [x/count for x in vecSum]
            else:
                muList[k] = 'Empty'

    return muList,ave_msd

def HAC(data, numClusters, metric):
    e = []
    for i in range(len(data)):
        e.append([data[i]])

    while len(e) > numClusters:
        # find two closest clusters
        minDist = sys.maxint
        a = []
        b = []
        for i in range(len(e)-1):
            for j in range(i+1, len(e)):
                currDist = metric(e[i], e[j], utils.squareDistance)
                if currDist < minDist:
                    minDist = currDist
                    a = e[i]
                    b = e[j]

        e.remove(a)
        e.remove(b)
        a.extend(b) # a is a cluster that now contains all the vectors that were in b

        e.append(a)

    return e


def AutoClass(data, numClusters):

    numFeatures = len(data[0])

    pi_k = [1.0/numClusters] * numClusters
    k_examples = []
    mu_k = []
    var_k = []
    en_k = []

    ####  Initialize the mu_k
    for i in range(numClusters):
        mu_k.append(data[random.randint(0, len(data))])
        k_examples.append([])

    #### Initialize variances to sample variance within groups
    #### First, assign data to clusters to get original estimate for sigma^2
    # for i in range(len(data)):
    #     distances = []
    #     for muVec in mu_k:
    #         if not(isinstance(muVec,str)):
    #             distances.append(utils.squareDistance(data[i], muVec))
    #     minDist = min(distances)
    #     k = distances.index(minDist)
    #     k_examples[k].append(data[i])
    # for k in range(numClusters):
    #     feat_k = zip(*k_examples[k])
    #     var_k.append(map(np.var,feat_k))
    var_k = [[0.3] * numFeatures] * numClusters

    log_lhood = -10000
    log_lhood_list = []
    prev_lhood = -100000
    epsilon = 0.01

    # Repeat until convergence
    iteration = 0
    while abs((log_lhood - prev_lhood)/log_lhood)>epsilon:
        # Expectation step
        print "Iteration: " + str(iteration)
        print pi_k
        print mu_k
        print var_k
        gam = [[0.0] * numClusters] * len(data)
        en_k = [0.0] * numClusters
        p_k = [0.0] * numClusters
        for n in range(len(data)):
            for k in range(numClusters):
                lhood = 1.0
                for d in range(numFeatures):
                    if var_k[k][d]==0:
                        lhood *= 1.0/math.sqrt(2*math.pi*0.0001)*math.exp(-(data[n][d]-mu_k[k][d])**2.0/(2.0*0.0001))
                    else:
                        lhood *= 1.0/math.sqrt(2*math.pi*var_k[k][d])*math.exp(-(data[n][d]-mu_k[k][d])**2.0/(2.0*var_k[k][d]))
                p_k[k] = pi_k[k] * lhood
            for k in range(numClusters):
                gam[n][k] = p_k[k]/sum(p_k)
                en_k[k] += gam[n][k]

        # Maximization step
        for k in range(numClusters): 
            for n in range(len(data)):
                for d in range(len(data[0])):
                    mu_k[k][d] += gam[n][k]*data[n][d]/en_k[k]
            temp_sum = [[0.0]*numFeatures]* numClusters
            for n in range(len(data)):
                for d in range(len(data[0])):
                    temp_sum[k][d] += gam[n][k]*(data[n][d]-mu_k[k][d])**2/en_k[k]
            for d in range(numFeatures):
                var_k[k][d] = temp_sum[k][d]/en_k[k]
            pi_k[k] = en_k[k]/sum(en_k)


        # Compute log-likelihood
        prev_lhood = log_lhood
        norm_sum = 0
        for n in range(len(data)):
            for k in range(numClusters):
                norm_prod = 1.0
                for d in range(len(mu_k)):
                    if var_k[k][d]==0:
                        norm_prod *= pi_k[k]*1.0/math.sqrt(2*math.pi*0.0001)*math.exp(-(data[n][d]-mu_k[k][d])**2.0/(2.0*0.0001))
                    else: 
                        norm_prod *= pi_k[k]*1.0/math.sqrt(2*math.pi*var_k[k][d])*math.exp(-(data[n][d]-mu_k[k][d])**2.0/(2.0*var_k[k][d]))
                norm_sum += pi_k[k]*norm_prod
            log_lhood += np.log(norm_sum)
        log_lhood_list.append(log_lhood)
        print log_lhood
        
        iteration+=1 

    return log_lhood_list,iteration


def scatterPlot3D(all_clus, metric):
    # Create Map
    from pylab import *

    col = ['red','green','blue','orange']

    ## Scatter plot of the instances in 3 dimensions
    plt.clf()
    fig = plt.figure()
    ax3D = fig.gca(projection='3d')
    for clus in range(len(all_clus)):
        x = []
        y = []
        z = []
        for i in range(len(all_clus[clus])):
            x.append(all_clus[clus][i][0])
            y.append(all_clus[clus][i][1])
            z.append(all_clus[clus][i][2])

            p3d = ax3D.scatter(x, y, z, c=col[clus], marker='o')   
    ax3D.set_xlabel('age')
    ax3D.set_ylabel('education')
    ax3D.set_zlabel('income')                                                                             
    savefig("4b"+metric+".png")

    plt.show()

    return



# main
# ----
# The main program loop
# You should modify this function to run your experiments

def main():
    # Validate the inputs
    if(validateInput() == False):
        print "Usage: clust numClusters numExamples"
        sys.exit(1);

    numClusters = int(sys.argv[1])
    numExamples = int(sys.argv[2])

    #Initialize the random seed
    
    random.seed()

    #Initialize the data

    
    dataset = file(DATAFILE, "r")
    if dataset == None:
        print "Unable to open data file"


    data = parseInput(dataset, numExamples)
    
    
    dataset.close()
    # printOutput(data,numExamples)

    # ==================== #
    # WRITE YOUR CODE HERE #
    # ==================== #

    ############################ FOR K-MEANS ##################################

    #### For running k-means once using the specified number of clusters in the command line ####
    # print "***********      K - MEANS WITH K = " + str(numClusters) + "       ***********"
    # muList, ave_msd = kMeans(data, numClusters)
    # for i in range(0,len(muList)):
    #     print "Center of cluster " + str(i+1) + " : " + str(muList[i])
    #     print "Average Mean Squared Distance: " + str(ave_msd) 
    # print "\n"

    #### For plotting k vs. MSE ####     
    # msd = []
    # for numClusters in range(2,11):
    #     muList, ave_msd = kMeans(data, numClusters)
    #     msd.append(ave_msd)
    #     print "***********      K - MEANS WITH K = " + str(numClusters) + "       ***********"
    #     for i in range(0,len(muList)):
    #         print "Center of cluster " + str(i+1) + " : " + str(muList[i])
    #         print "\n"
    #     print "Average Mean Squared Distance: " + str(ave_msd) 
    #     print "\n\n"

    # plt.clf()
    # xs = range(2,11)
    # ys = msd

    # pl=plt.plot(xs,ys,color='b')

    # plt.title("Graph 4(a)(a) Plot of K vs. MSE")
    # plt.xlabel('k = number of clusters')
    # plt.ylabel('mean squared error')

    # plt.show()    

    ############################ FOR HAC ALGORITHM ################################## 
    # dataset = file("adults-small.txt", "r")
    # if dataset == None:
    #     print "Unable to open data file"

    # data_small = parseInput(dataset, numExamples)
    
    # dataset.close()

    # print "***********      HAC ALGORITHM = " + str(numClusters) + "       ***********"
    # if(len(data_small)==100):
    #     all_clus_min = HAC(data_small, numClusters, utils.cmin)
    #     inst_per_clus_min = [len(all_clus_min[i]) for i in range(len(all_clus_min))] 
    #     print inst_per_clus_min

    #     scatterPlot3D(all_clus_min, "minimum")

    #     all_clus_max = HAC(data_small, numClusters, utils.cmax)
    #     inst_per_clus_max = [len(all_clus_max[i]) for i in range(len(all_clus_max))] 

    #     print inst_per_clus_max

    #     scatterPlot3D(all_clus_max, "maximum")

    # if(len(data_small)==200):
    #     all_clus_cmean = HAC(data_small, numClusters, utils.cmean)
    #     inst_per_clus_cmean = [len(all_clus_cmean[i]) for i in range(len(all_clus_cmean))] 
    #     print inst_per_clus_cmean

    #     scatterPlot3D(all_clus_cmean, "mean")

    #     all_clus_ccent = HAC(data_small, numClusters, utils.ccent)
    #     inst_per_clus_ccent = [len(all_clus_ccent[i]) for i in range(len(all_clus_ccent))] 

    #     print inst_per_clus_ccent

    #     scatterPlot3D(all_clus_ccent, "centroid")

    ############################ FOR AUTOCLASS ##################################

    dataset = file("adults-small.txt", "r")
    if dataset == None:
        print "Unable to open data file"

    data = parseInput(dataset, numExamples)
    
    dataset.close()

    #### For running k-means once using the specified number of clusters in the command line ####
    print "***********      AUTO-CLASS WITH K = " + str(numClusters) + "       ***********"
    lhood, iteration = AutoClass(data, numClusters)
    for i in range(1,(len(lhood)+1)):
        print "Log likelihood of iteration " + str(i-1) + ": " + str(lhood[i-1])
    print "\n"

    #### For plotting k vs. MSE ####     
    # msd = []
    # for numClusters in range(2,11):
    #     muList, ave_msd = kMeans(data, numClusters)
    #     msd.append(ave_msd)
    #     print "***********      K - MEANS WITH K = " + str(numClusters) + "       ***********"
    #     for i in range(0,len(muList)):
    #         print "Center of cluster " + str(i+1) + " : " + str(muList[i])
    #         print "\n"
    #     print "Average Mean Squared Distance: " + str(ave_msd) 
    #     print "\n\n"

    # plt.clf()
    # xs = range(2,11)
    # ys = msd

    # pl=plt.plot(xs,ys,color='b')

    # plt.title("Graph 4(a)(a) Plot of K vs. MSE")
    # plt.xlabel('k = number of clusters')
    # plt.ylabel('mean squared error')

    # plt.show()  

if __name__ == "__main__":
    validateInput()
    main()
