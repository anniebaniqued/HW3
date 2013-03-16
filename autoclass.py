
def AutoClass(data, numClusters):

    numFeatures = len(data[0])

    pi_k = [1/numClusters] * numClusters
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
    for i in range(len(data)):
        distances = []
        for muVec in muList:
            if not(isinstance(muVec,str)):
                distances.append(utils.squareDistance(data[i], muVec))
        minDist = min(distances)
        k = distances.index(minDist)
        k_examples[k].append(data[i])
    for k in range(numClusters):
        feat_k = zip(*k_examples[k])
        var_k.append(map(np.var,feat_k))

    log_lhood = 0
    prev_lhood = -100000
    epsilon = 1

    # Repeat until convergence
    while abs((log_lhood - prev_log_lhood)/log_lhood)>epsilon:
        # Expectation step
        gam_n_k = [[0.0] * len(data)] * numClusters
        en_k = [0.0] * numClusters
        for n in range(len(data)):
            p_k = []*numClusters
            for k in range(numClusters):
                lhood = 0.0
                for d in range(len(mu_k)):
                    lhood *= 1.0/sqrt(2*pi*var_k[k][d])*exp(-(data[n][d]-mu_k[k][d])**2.0/(2.0*var_k[k][d]**2))
                p_k[k] = pi_k[k] * lhood
            for k in range(numClusters):
                gam_n_k[k][n] = pi_k[k]/sum(pi_k)
                en_k[k] += gam_n_k

        # Maximization step
        mu_k = [[0] * numFeatures] * numClusters
        for k in range(numClusters): 
            for n in range(len(data)):
                for d in range(len(data[0])):
                    mu_k[k][d] += gam_n_k[k][n]*data[n][k]/sum(gam_n_k[k])
            temp_sum = [[0.0]*numFeatures]* numClusters
            for n in range(len(data)):
                for d in range(len(data[0])):
                    temp_sum[k][d] += gam_n_k[k][n]*(data[n][d]-mu_k[k][d])**2/sum(gam_n_k[k])
            var_k[k] = temp_sum[k][d]/sum(gam_n_k[k])
            pi_k[k] = en_k[k]/sum(en_k)


        # Compute log-likelihood
        prev_lhood = log_lhood
        for n in range(len(data)):
            for k in range(numClusters):
                for d in range(len(mu_k)):
                    norm_sum += 1.0/sqrt(2*pi*var_k[k][d])*exp(-(data[n][d]-mu_k[k][d])**2.0/(2.0*var_k[k][d]**2))
            log_lhood += np.log(norm_sum)
    



