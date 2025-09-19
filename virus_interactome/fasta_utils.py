
def change_id_proteome (inputpath, outputpath):
    
  '''Define a function in order to clean a .txt file
  ARGS IN: inputpath: path to the input .txt file
            outputpath: path to the output .txt file
  '''

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
            protein_id = line[start+8:end].replace(' ','_') 
            #import pdb; pdb.set_trace()
            newline = f">{protein_id}|{line[1:]}"
            outputfile.write(newline)

        else:
            outputfile.write(line)
