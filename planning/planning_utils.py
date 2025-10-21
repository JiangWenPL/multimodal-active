import numpy as np
import matplotlib as mpl

color_mapping_3 = {
    0: np.array([255,255,255]), # white
    1: np.array([0,0,255]), # blue
    2: np.array([0,255,0]), # green
}

# plasma color map
heatmap = mpl.cm.get_cmap('plasma')

class LocalizationError(Exception):
    pass

def combimed_heuristic(unique_label, counts, labels, cam_pos):
    max_score = 0
    combined_target_label = -1
    
    for label, count in zip(unique_label, counts):
        frontier_pos = np.stack(np.where(labels == label), axis=1) # (K, 2)
        if len(frontier_pos) < 4:
            continue
        
        frontier_distance = np.linalg.norm(frontier_pos - cam_pos, axis=1)
        mean_distance = frontier_distance.mean()
        area_score = (count) / (mean_distance + 20)

        if area_score > max_score:
            max_score = area_score
            combined_target_label = label
    return combined_target_label