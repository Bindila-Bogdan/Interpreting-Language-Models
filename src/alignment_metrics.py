import numpy as np


def compute_mean_dot_product(explanations, known_evidences):
    """
    Computes the average dot product between the explanation and known
    evidence over all sentences.
    """

    sum_of_dot_products = 0

    for explanation, known_evidence in zip(explanations, known_evidences):
        sum_of_dot_products += np.dot(explanation, known_evidence)

    average_dot_product = sum_of_dot_products / len(explanations)

    return average_dot_product


def compute_probes_needed(explanation, known_evidence):
    """
    Computes the number of words we need to probe (look at) based on
    the saliency map to find the most important word. Basically, we
    get the rank of the word of interest after words were sorted
    descending by the saliency value.
    """

    probes_number = 1
    explanation_evidence = dict(zip(explanation, known_evidence)).items()
    sorted_explanation_evidence = sorted(explanation_evidence, reverse=True)

    for word_info in sorted_explanation_evidence:
        saliency_value, is_known_evidence = word_info

        if is_known_evidence == 0:
            probes_number += 1
        else:
            break

    return probes_number


def compute_mean_probes_needed(explanations, known_evidences):
    """
    Computes the average number of words we need to probe (look at) based on
    the saliency map to find the most important word. Basically, we get the
    average rank of the word of interest after words were sorted descending
    by the saliency value.
    """

    sum_of_probes_needed = 0

    for explanation, known_evidence in zip(explanations, known_evidences):
        sum_of_probes_needed += compute_probes_needed(explanation, known_evidence)

    average_probes_needed = sum_of_probes_needed / len(explanations)

    return average_probes_needed


def compute_mean_reciprocal_rank(explanations, known_evidences):
    """
    Calculates the average of the inverse rank of the first token that
    is part of the known evidence.
    """

    sum_of_inverse_ranks = 0

    for explanation, known_evidence in zip(explanations, known_evidences):
        # consider only the appearance of the first token of interest
        for index in np.where(known_evidence == 1)[0][1:]:
            known_evidence[index] = 0

        inverse_ranking = 1 / compute_probes_needed(explanation, known_evidence)
        sum_of_inverse_ranks += inverse_ranking

    mean_inverse_rank = sum_of_inverse_ranks / len(explanations)

    return mean_inverse_rank