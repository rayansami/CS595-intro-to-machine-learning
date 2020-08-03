import nbformat
import nbconvert.exporters as ex
import sys
import os.path

def pprint(pdf_file, source_files):
    
    cells = []
    for filename in source_files:

        if filename.endswith('.py'):
            with open(filename, 'r') as f:
                code_src = f.read()

            code_cell = nbformat.v4.new_code_cell(source=code_src, execution_count=1, outputs=[])

            if len(cells) > 0:
                cells += [nbformat.v4.new_markdown_cell( source = r'\newpage \setcounter{section}{0}' )]

            cells += [
                nbformat.v4.new_markdown_cell( source = '# CODE LISTING: ``' + filename + '``' ),
                code_cell
            ]

        elif filename.endswith('.ipynb'):

            this_notebook = nbformat.read(filename, 4)

            if len(cells) > 0:
                cells += [nbformat.v4.new_markdown_cell( source = r'\newpage \setcounter{section}{0}' )]

            cells += [nbformat.v4.new_markdown_cell( source = '# NOTEBOOK LISTING: ``' + filename + '``' )]
            cells += this_notebook.cells

        else:
            raise ValueError('Unrecognized format: ' + filename)

    # Export notebook
    notebook = nbformat.v4.new_notebook(cells=cells)

    # Converter
    pdf_exporter = ex.PDFExporter(latex_count=1)

    # Directory of this script
    dir_path = os.path.dirname(os.path.realpath(__file__))    
    pdf_exporter.template_file = dir_path + '/pdf.tplx'

    # Convert
    output, resources = pdf_exporter.from_notebook_node(notebook)

    # Write to file
    with open(pdf_file, 'wb') as f:
        f.write(output)
