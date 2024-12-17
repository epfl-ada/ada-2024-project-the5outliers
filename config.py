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
    'World Regions': '#1AC5BD',
    'Others': '#696e79'
}

# Generate shades of grey for remaining categories
NUM_GREYS = len(CATEGORIES_OTHERS) - len(HIGHLIGHT_COLORS)  # Excluding highlighted categories
grey_shades = [mcolors.to_hex((v, v, v)) for v in np.linspace(0.2, 0.4, NUM_GREYS)]
non_custom_categories = [cat for cat in CATEGORIES_OTHERS if cat not in HIGHLIGHT_COLORS]
GREY_PALETTE = dict(zip(non_custom_categories, grey_shades))

# Combine highlight colors and grey palette
PALETTE_CATEGORY_DICT = {**HIGHLIGHT_COLORS, **GREY_PALETTE}



# Generate darker color shades for remaining categories
NUM_DARKER = len(CATEGORIES_OTHERS) - len(HIGHLIGHT_COLORS)  # Excluding highlighted categories
# fixed Saturation and Brightness
S = 0.44  # 44% saturation
B = 0.6  # 64% brightness
# Generate 15 evenly spaced Hue values from 0° to 223°
hue_values = np.linspace(191, 326, NUM_DARKER) / 360  # Convert degrees to [0, 1] range
# Convert HSB (or HSV) to RGB and then to HEX
darker_shades = [mcolors.to_hex(mcolors.hsv_to_rgb((H, S, B))) for H in hue_values]
non_custom_categories = [cat for cat in CATEGORIES_OTHERS if cat not in HIGHLIGHT_COLORS]
DARKER_PALETTE = dict(zip(non_custom_categories, darker_shades))

# Combine highlight colors and grey palette
PALETTE_CATEGORY_DICT_COLORS = {**HIGHLIGHT_COLORS, **DARKER_PALETTE}