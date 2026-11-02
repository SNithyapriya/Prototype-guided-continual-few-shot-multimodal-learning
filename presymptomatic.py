def infection_risk(healthy_probability):
    return 1.0-healthy_probability

def classify_presymptomatic(risk,threshold=0.5):
    return int(risk>=threshold)
