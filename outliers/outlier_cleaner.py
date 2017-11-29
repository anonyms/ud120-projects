#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
        
    cleaned_data = []

    ### your code goes here
    from numpy import dstack
    errors = abs(predictions-net_worths)
    cleaned_data = dstack((ages,net_worths,errors))
    cleaned_data = cleaned_data.tolist()
    for i in range(0,len(cleaned_data)):
        cleaned_data[i] = cleaned_data[i][0]
    cleaned_data = [tuple(l) for l in cleaned_data]


    cleaned_data.sort(key=lambda tup:tup[2])

    minus_10_percent = int(round(len(cleaned_data)*0.9))
    cleaned_data = cleaned_data[0:minus_10_percent]

    return cleaned_data

