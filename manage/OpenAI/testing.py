import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_clipboard()

plt.hist(data['Home Sales'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Home Sales')
plt.xlabel('Home Sales')
plt.ylabel('Frequency')
plt.savefig('graph.png')
plt.show()
