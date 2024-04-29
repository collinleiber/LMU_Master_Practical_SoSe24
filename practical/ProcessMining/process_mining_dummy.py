import itertools
import random
import math
import time
import matplotlib.pyplot as plt


def kMeans (data,k):

    #initialise centroids
    centroids = []
    clusters =[]

    for _ in range(k):
        random_point =random.choice(data)
        while random_point in centroids:
            random_point= random.choice(data)
        centroids.append(random.choice(data)) 
        clusters.append([])
        print(centroids)

        
    while True:

        old_clusters=clusters.copy()
        clusters = []
        for _ in range(k):
            clusters.append([])
        #assignment phase
        for point in data:
        



            nearest_centroid = None
            distance = float('inf')

            for index, centroid in enumerate(centroids):
                if distance > math.dist(point,centroid):
                    distance = math.dist(point,centroid)
                    nearest_centroid = index
            

            clusters[nearest_centroid].append(point)

       
    
    #update phase
        for index,cluster in enumerate(clusters) :
            count_points = len(cluster)
            sum_x = sum([tupel[0] for tupel in cluster])
            sum_y = sum([tupel[1] for tupel in cluster])
            avg_x= sum_x/count_points
            avg_y= sum_y/count_points
            centroids [index] = (avg_x,avg_y)
        
        print ("Clusters {}".format(clusters))
        print ("Old Clusters {}".format(old_clusters))
        print ("Centroids {}".format(centroids))

        time.sleep(1)




        if clusters == old_clusters:
            #generate_plot(clusters,k)
            return clusters
        


def generate_plot (clusters,k):
    colors = itertools.cycle(["r", "b", "g"])
    for i in range(k):
        plt.scatter([tupel[0] for tupel in clusters[i]],[tupel[1] for tupel in clusters[i]],color=next(colors))
    plt.show() 
    
kMeans([(2,3),(4,8),(8,7),(10,3),(5,7),(1,3),(3,11),(9,2),(8,4),(5,1)],2)      



