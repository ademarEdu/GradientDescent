
def queryFunction(functions):
    """
    This function asks the user to select a function from the list of functions and returns the selected function.
     
    Args:
        functions (list): A list of functions to choose from.    

    Returns:
        function (object): The selected function object.
    """
    # Show the user the list of functions to choose from
    print("Seleccione una función:")
    for i, func in enumerate(functions):
        print(f"{i + 1}. {func}")

    # Ask the user to select a function
    choice = int(input("Ingrese el número de la función: "))
    while choice < 1 or choice > len(functions):
        print("Opción no válida. Intente de nuevo.")
        choice = int(input("Ingrese el número de la función: "))

    match(choice):
        case 1:
            from functions.sphere import Sphere
            function = Sphere()
        case 2:
            from functions.cigar import Cigar
            function = Cigar()
        case 3:
            from functions.rosenbrock import Rosenbrock
            function = Rosenbrock()

    return function