import torch


def minimum_distance_regularization(state_matrix, min_distance, p=2):
    """
    Compute a regularization loss based on minimum distance between the rows of the matrix.

    :param state_matrix: [num_states, output_size] Tensor, represents the states
    :param min_distance: Scalar, minimum allowed distance between states
    :param p: Scalar, the norm to be used in the distance computation
    :return: Scalar tensor, the regularization loss
    """
    num_states = state_matrix.size(0)

    # Compute pairwise distances using torch.cdist
    pairwise_distances = torch.cdist(state_matrix, state_matrix, p=p)

    # Create a mask to ignore self-distances (distance from a point to itself)
    mask = torch.eye(num_states, device=state_matrix.device).bool()

    # Use where to avoid in-place modification of pairwise_distances
    pairwise_distances = torch.where(mask, torch.tensor(float('inf'), device=state_matrix.device), pairwise_distances)

    #print(state_matrix)

    # Identify pairs of points that are too close and compute their violation
    violations = torch.clamp(min_distance - pairwise_distances, min=0)

    # Sum all violations to get the regularization loss
    loss = violations.sum()

    return loss

def get_upper_triangular_distances(state_matrix, p=2):
    num_states = state_matrix.size(0)

    # Compute pairwise squared Euclidean distances
    pairwise_distances = torch.cdist(state_matrix, state_matrix, p=p)

    # Create a mask to ignore self-distances and lower triangular part
    mask = torch.triu(torch.ones(num_states, num_states), diagonal=1).bool().to(state_matrix.device)

    # Use the mask to extract the upper triangular part (excluding the diagonal)
    pairwise_distances = pairwise_distances[mask]

    return pairwise_distances
