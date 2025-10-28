
def change_id_proteome (inputpath, outputpath):
    
  """
  Cleans and reformats a proteome file by extracting and modifying protein identifiers.

  This function reads a file line by line, identifies header lines starting with '>',
  extracts the protein name from the 'protein=' field, and replaces the original header
  with a standardized format. The cleaned content is written to a new output file.

  Parameters
  ----------
  inputpath : str
      Path to the input file containing raw proteome data.
  outputpath : str
      Path to the output file where the cleaned data will be saved.

  Returns
  -------
  None

  Raises
  ------
  FileNotFoundError
      If the input file does not exist.
  ValueError
      If the protein name cannot be extracted from a header line.
  """

  with open (inputpath,'r') as inputfile:
    with open (outputpath,'w') as outputfile:
      #Buscar linea que comience con >
      for line in inputfile:
        if line.startswith('>'):
          #Encontrar la primera ocurrencia de proteina = 'nombre  
          # de la proteina' y extraer el nombre de la proteina
          start = line.find('protein=')
          if start != -1:
            end = line.find(']', start)
            protein_id = line[start+8:end].replace(' ','_').replace('.','_').replace('/','_').replace('.','_')
            #import pdb; pdb.set_trace()
            newline = f">{protein_id}|{line[1:]}"
            outputfile.write(newline)

        else:
            outputfile.write(line)
