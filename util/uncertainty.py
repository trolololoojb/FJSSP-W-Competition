import random

def create_uncertainty_vector(n_required, factor : float = 10.0, offset : float = 1.0):
    """
    Args: 
        n_required: Number of uncertainty parameters to generate.
        factor: A multiplier for the alpha parameter to create the beta parameter.
        offset: A constant value added to both alpha and beta parameters to ensure they are not too small.
    Returns:
        A list of uncertainty parameters, where each parameter is a list containing alpha, beta, and offset values.
    """
    uncertainty_parameters = []
    for _ in range(n_required):
        alpha = random.random()
        beta = factor * alpha
        offset = offset
        uncertainty_parameters.append([alpha, beta, offset])
    return uncertainty_parameters