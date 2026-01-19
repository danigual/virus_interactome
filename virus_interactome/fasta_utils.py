import os

def change_id_proteome (inputpath: str, outputpath: str)-> None:
    
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

  """
  if not os.path.exists(inputpath):
    raise FileNotFoundError(f"Input file not found: {inputpath}")
    
  with open (inputpath,'r') as inputfile:
    
    with open (outputpath,'w') as outputfile:
      
      for line in inputfile:
        if line.startswith('>'):
          
          start = line.find('protein=')
          if start != -1:
            end = line.find(']', start)
            protein_id = line[start+8:end].replace(' ','_').replace('.','_').replace('/','_')
            
            newline = f">{protein_id}|{line[1:]}"
            outputfile.write(newline)

        else:
            outputfile.write(line)
