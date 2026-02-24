def generate_table(name, dimension, data, output_path):
    """
    This function generates a table displaying the results of the gradient descent.

    Args:
    name (str): The name of the function being optimized (e.g. "Sphere", "Cigar", "Rosenbrock").
    dimension (int): The dimension of the function being optimized.
    data (np array): A 10x3 array where each row contains the alpha value, mean squared error, and performance for a specific alpha value.
    output_path (str): The file path where the generated LaTeX table will be saved.
    """
    # LaTeX template for the table
    template = f"""\n
\\begin{{table}}[h]
    \\centering
    \\caption{{{name}, {dimension} dimensiones}}
    \\vspace{{2mm}}
    \\begin{{tabular}}{{r|ccc}}
        \\hline
        $\\alpha$ & Accuracy & Performance \\\\ [0.5ex]
        \\hline
table_data
        \\hline
    \\end{{tabular}}
\\end{{table}}
"""

    # Generate the rows of the table
    rows = ""
    for alpha, mse, performance in data:
        rows += f"        {alpha:.2e} & {mse:.2e} & {performance:.2f} \\\\\n"
    
    # Insert the rows into the template
    table = template.replace("table_data", rows)

    # Add the table to the output file
    with open(output_path, "a") as f:
        f.write(table)