import matplotlib.pyplot as plt
import pandas as pd

# Prepare the data
data = {
    'App': [
        'addressbook', 'claroline', 'ppma', 'mrbs', 'mantisbt',
        'dimeshift', 'pagekit', 'phoenix', 'petclinic'
    ],
    'BERT + Siamese (BCE)': [119, 24, 9, 14, 69, 10, 40, 5, 2],
    'FRAGGEN': [39, 55, 22, 22, 45, 71, 102, 16, 10],
    'WEBEMBED': [27, 17, 7, 27, 101, 31, 13, 21, 4],
    'RTED': [22, 47, 5, 17, 53, 5, 37, 14, 13],
    'PDiff': [125, 31, 5, 33, 9, 5, 36, 14, 4]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set 'App' as the index so it's used as the x-axis
df.set_index('App', inplace=True)

# Plot each column (method) as a bar chart
ax = df.plot(kind='bar', figsize=(10, 6))

# Set chart labels and title
ax.set_xlabel('App')
ax.set_ylabel('Number of States of the Method')
ax.set_title('Number of States by App for Each Method')

# Show the legend and plot
plt.legend(title='Methods')
plt.tight_layout()
plt.show()
