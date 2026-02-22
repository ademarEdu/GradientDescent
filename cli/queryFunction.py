from gradient_descent import GD
import numpy as np

def queryFunction():
    """
    This function asks the user to select a function from the list of functions and returns the selected function.
     
    Args:
        functions (list): A list of functions to choose from.    

    Returns:
        optimizer (GD): The Gradient Descent optimizer object initialized with the selected function.
        function (object): The selected function object.
    """
    # Show the user the list of functions to choose from
    functions = ["Sphere", "Cigar", "Rosenbrock"]

    print("\nSeleccione una función:")
    for i, func in enumerate(functions):
        print(f"{i + 1}. {func}")
    print(f"{len(functions) + 1}. Salir")

    # Ask the user to select a function
    choice = int(input("\nIngrese el número de la función: "))
    while choice < 1 or choice > len(functions) + 1:
        print("Opción no válida. Intente de nuevo.")
        choice = int(input("Ingrese el número de la función: "))

    match(choice):
        case 1:
            from functions.sphere import Sphere
            function = Sphere(2)
            optimizer = GD(function, 0.25, 1000)

        case 2:
            from functions.cigar import Cigar
            function = Cigar(2)
            optimizer = GD(function, 0.1e-7, 100000)
        case 3:
            from functions.rosenbrock import Rosenbrock
            function = Rosenbrock(2)
            optimizer = GD(function, 0.001, 5000)
        case 4:
            print("Saliendo del programa.")
            exit()

    return optimizer, function