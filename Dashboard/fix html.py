import re

with open("charts.html", "r", encoding="utf-8") as file:
    content = file.read()

# Replace all static/dashboard/xyz references
pattern = r'(["\'\(=]\s*)static/dashboard/(.*?\.(css|js|png|jpg|jpeg|svg|ico))'
content = re.sub(pattern, r'\1{% static \'dashboard/\2\' %}', content)

with open("your_file_fixed.html", "w", encoding="utf-8") as file:
    file.write(content)

print("✅ All static paths converted.")
