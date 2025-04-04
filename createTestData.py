import random

"""
File creates some random test and training data for .txt file in correct form that can be used in testing/developing ANN

"""


def testDataFunction(params):
    p1 = params[0] + params[1] + params[2] + params[3]
    p2 = 3*params[0] + 3*params[1] - 5 * params[2] + params[3] + 18
    #p1 = params[0]*2 + 3
    #p2 = params[0] * 180.23 - 65
    return [p1, p2]

def generateTrainingData(filename, num_samples=1000, num_params=1, num_results=2):
    """
    Luo koulutusdataa ja tallentaa sen tiedostoon.
    """
    with open(filename, 'w') as file:
        for _ in range(num_samples):
            params = [round(random.uniform(-100, 100), 4) for _ in range(num_params)]
            results = testDataFunction(params)
            param_str = " ".join(map(str, params))
            result_str = " ".join(map(str, results))
            file.write(f"{param_str} | {result_str}\n")

if __name__ == "__main__":
    generateTrainingData("training_data.txt", num_samples=1000, num_params=4, num_results=2)
    print("Skripti suoritettu")
