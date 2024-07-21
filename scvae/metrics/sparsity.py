import numpy as np

def calculate_sparsity(W):
    D = np.shape(W)[0]
    numerator = np.sum( np.abs(W), axis=0)
    denominator = np.sqrt(np.sum(np.power(W,2),axis=0))
    subtract_by = np.divide(numerator, denominator)
    del numerator, denominator
    subtract_from = np.sqrt(D)
    subtracted = subtract_from - subtract_by
    del subtract_from, subtract_by
    numerator = np.mean(subtracted)
    del subtracted
    denominator = np.sqrt(D)-1
    sparsity = np.divide( numerator, denominator)
    return sparsity


def look_sparsity(z):

    total_number = z.numel()
    total_nonzero = z.count_nonzero()
    sparsity_nonzero = (total_number-total_nonzero)/total_number

    sparsity_nmf = calculate_sparsity(z.detach().cpu().numpy().reshape(-1))

    return sparsity_nonzero, sparsity_nmf