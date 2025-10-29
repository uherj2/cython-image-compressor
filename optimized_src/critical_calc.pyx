import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel

ctypedef np.uint8_t DTYPE_UINT8
ctypedef np.intp_t DTYPE_INTP  
ctypedef np.float64_t DTYPE_FLOAT 

@cython.boundscheck(False)  
@cython.wraparound(False)
cpdef initialize_centroids(np.ndarray[DTYPE_UINT8, ndim=2] data, int k):
    cdef np.ndarray centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    return centroids.astype(np.float64)

# Helper function to compute squared distance (avoids reduction issue)
@cython.boundscheck(False)
cdef inline DTYPE_FLOAT squared_distance(DTYPE_UINT8[:] point, DTYPE_FLOAT[:] centroid, DTYPE_INTP M) nogil:
    cdef DTYPE_FLOAT sq_dist = 0.0
    cdef DTYPE_FLOAT diff
    cdef DTYPE_INTP m
    for m in range(M):
        diff = <DTYPE_FLOAT>point[m] - centroid[m]
        sq_dist += diff * diff
    return sq_dist

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef assign_clusters(np.ndarray[DTYPE_UINT8, ndim=2] data_in, np.ndarray[DTYPE_FLOAT, ndim=2] centroids_in):
    # MEMORY VIEWS
    cdef DTYPE_UINT8[:, :] data = data_in
    cdef DTYPE_FLOAT[:, :] centroids = centroids_in

    cdef DTYPE_INTP N = data.shape[0]        
    cdef DTYPE_INTP K = centroids.shape[0]   
    cdef DTYPE_INTP M = data.shape[1]        

    # OUT ARRAY
    cdef np.ndarray[DTYPE_UINT8, ndim=1] labels_out = np.empty(N, dtype=np.uint8)
    cdef DTYPE_UINT8[:] labels = labels_out

    # declared before loop because of nogil
    cdef DTYPE_INTP i, j, m # Loop indices
    cdef DTYPE_FLOAT current_sq_dist 
    cdef DTYPE_FLOAT diff            
    cdef DTYPE_FLOAT min_sq_dist
    cdef DTYPE_UINT8 min_index

    for i in prange(N, schedule='static', chunksize=32, nogil=True): 
        min_index = 0

        min_sq_dist = squared_distance(data[i, :], centroids[0, :], M)
        for j in range(1, K):
            current_sq_dist = squared_distance(data[i, :], centroids[j, :], M)

            if  current_sq_dist < min_sq_dist:
                min_sq_dist = current_sq_dist
                min_index = <DTYPE_UINT8> j

        labels[i] = min_index
    
    return labels_out

def update_centroids(data, labels, k):
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])

def apply_label(label, centroids):
    return centroids[label]
