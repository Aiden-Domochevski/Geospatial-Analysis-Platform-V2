"""Assemble both parts and inject final Chapter 4 into rapport.tex"""
import os

# Read parts
with open("ch4_part1.tex", "r", encoding="utf-8") as f:
    part1 = f.read()
with open("ch4_part2.tex", "r", encoding="utf-8") as f:
    part2 = f.read()

full_chapter = part1 + "\n" + part2

# Read rapport.tex
tex_path = os.path.join("..", "documentation", "rapport.tex")
with open(tex_path, "r", encoding="utf-8") as f:
    content = f.read()

# Find where Chapter 4 starts
marker = "\\chapter{Implémentation Technique, Modélisation Appliquée et Benchmarking}"
end_marker = "\\end{document}"

if marker in content:
    # Replace everything from Chapter 4 to end
    before = content.split(marker)[0]
    new_content = before + full_chapter + "\n\n" + end_marker + "\n"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Chapter 4 replaced successfully!")
    print(f"Total rapport length: {len(new_content)} characters")
    print(f"Chapter 4 content: {len(full_chapter)} characters (~{len(full_chapter)//2000} pages estimated)")
else:
    print(f"ERROR: Could not find chapter marker in rapport.tex")
