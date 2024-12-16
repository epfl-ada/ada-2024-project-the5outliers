import numpy as np
import matplotlib.colors as mcolors

# Category abbreviations
CATEGORY_ABBREVIATIONS = {
    'Art': 'Art',
    'Mathematics': 'Math',
    'IT': 'IT',
    'Business Studies': 'BS',
    'Music': 'Music',
    'Religion': 'R',
    'Language and literature': 'L&L',
    'Citizenship': 'CIT',
    'World Regions': 'WR',
    'Design and Technology': 'D&T',
    'Everyday life': 'Life',
    'History': 'Hist',
    'People': 'P',
    'Geography': 'Geo',
    'Science': 'Sci'
}

# Categories including "Others"
CATEGORIES_OTHERS = [
    'Art',
    'Business Studies',
    'Citizenship',
    'World Regions',
    'Design and Technology',
    'Everyday life',
    'Geography',
    'History',
    'IT',
    'Language and literature',
    'Mathematics',
    'Music',
    'People',
    'Religion',
    'Science',
    'Others'
]

# Highlight colors for specific categories
HIGHLIGHT_COLORS = {
    'World Regions': '#2CB5AE'
}

# Generate shades of grey for remaining categories
NUM_GREYS = len(CATEGORIES_OTHERS) - len(HIGHLIGHT_COLORS)  # Excluding highlighted categories
grey_shades = [mcolors.to_hex((v, v, v)) for v in np.linspace(0.2, 0.4, NUM_GREYS)]
non_custom_categories = [cat for cat in CATEGORIES_OTHERS if cat not in HIGHLIGHT_COLORS]
GREY_PALETTE = dict(zip(non_custom_categories, grey_shades))

# Combine highlight colors and grey palette
PALETTE_CATEGORY_DICT = {**HIGHLIGHT_COLORS, **GREY_PALETTE}
