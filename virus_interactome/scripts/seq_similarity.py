from Bio import SeqIO, pairwise2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sequences = list(SeqIO.parse("/home/daniel/ppi_data_remote/adeno/0_proteomes/curated/HAdV5_AC_000008_1_modified.fa", "fasta"))
n = len(sequences)
matrix = np.zeros((n, n))
labels = [seq.id for seq in sequences]


for i in range(n):
    for j in range(n):
        alignments = pairwise2.align.globalxx(sequences[i].seq, sequences[j].seq)
        score = alignments[0].score
        max_len = max(len(sequences[i].seq), len(sequences[j].seq))
        matrix[i][j] = score / max_len 

# Crear el heatmap
plt.figure(figsize=(14, 14))
sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", annot=False)
plt.title("Similitud entre proteínas del proteoma de AdV5")
plt.tight_layout()
plt.show()