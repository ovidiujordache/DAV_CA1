from __init__ import *



# URL after modification for direct download
url = 'https://studentdkit-my.sharepoint.com/:x:/g/personal/d00187472_student_dkit_ie/EUmxpdo-669JvlV-MrVH2S0BPyfqdJmUVRsUtaafssdExg?e=po7t3d'

# Read CSV file into DataFrame
df = pd.read_csv(url)

# Display the DataFrame
print(df.head())
