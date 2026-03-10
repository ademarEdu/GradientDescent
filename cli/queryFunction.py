from gradient_descent import GD
import numpy as np

def queryCondition(conditions):
    """
    This function asks the user to choose a condition for the optimization process from the list of conditions and returns the string selected.

    Args:
        conditions (list): A list of conditions to choose from. The list must contain strings of all the options.

    Returns:
        condition_choice (str): The string corresponding to the selected condition.
    """
    # Show the user the list of conditions to choose from
    print("\nSeleccione una condicion:")
    for i, condition in enumerate(conditions):
        print(f"{i + 1}. {condition}")

    # Ask the user to select a condition
    condition_choice = int(input("\nIngrese el número del condición: "))
    while condition_choice < 1 or condition_choice > len(conditions):
        print("Opción no válida. Intente de nuevo.")
        condition_choice = int(input("Ingrese el número del método: "))

    return conditions[condition_choice - 1]

def queryMethod(methods):
    """
    This function asks the user to select an optimization method from the list of methods and returns the selected method.

    Args:
        methods (list): A list of optimization methods to choose from.

    Returns:
        method_choice (int): The number corresponding to the selected optimization method.
    """
    # Show the user the list of optimization methods to choose from
    print("\nSeleccione un método de optimización:")
    for i, method in enumerate(methods):
        print(f"{i + 1}. {method}")

    # Ask the user to select an optimization method
    method_choice = int(input("\nIngrese el número del método: "))
    while method_choice < 1 or method_choice > len(methods):
        print("Opción no válida. Intente de nuevo.")
        method_choice = int(input("Ingrese el número del método: "))

    return method_choice

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
    functions = ["Sphere", "Cigar", "Rosenbrock", "Griewangk"]
    methods = ["Negative Gradient", "Newton"]
    conditions = ["Armijo", "Sufficient Decrease", "Curvature", "Strong Wolfe", "Goldstein"]

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
            method_choice = queryMethod(methods)
            condition_choice = queryCondition(conditions)
            optimizer = GD(function, 0.5, 1000, method=methods[method_choice - 1], condition=condition_choice)
        case 2:
            from functions.cigar import Cigar
            function = Cigar(2)
            method_choice = queryMethod(methods)
            condition_choice = queryCondition(conditions)
            optimizer = GD(function, 0.5, 10000, method=methods[method_choice - 1], condition=condition_choice)
        case 3:
            from functions.rosenbrock import Rosenbrock
            function = Rosenbrock(2)
            method_choice = queryMethod(methods)
            condition_choice = queryCondition(conditions)
            optimizer = GD(function, 0.1, 5000, method=methods[method_choice - 1], condition=condition_choice)
        case 4:
            from functions.griewangk import Griewangk
            function = Griewangk(2)
            method_choice = queryMethod(methods)
            condition_choice = queryCondition(conditions)
            optimizer = GD(function, 3, 10000, method=methods[method_choice - 1], condition=condition_choice)
        case 5:
            print("Saliendo del programa.")
            exit()

    return optimizer, function