"""Assemble Chapter 5 parts and inject into rapport.tex"""
import os

with open("ch5_part1.tex", "r", encoding="utf-8") as f:
    part1 = f.read()
with open("ch5_part2.tex", "r", encoding="utf-8") as f:
    part2 = f.read()

full_chapter = part1 + "\n" + part2

tex_path = os.path.join("..", "documentation", "rapport.tex")
with open(tex_path, "r", encoding="utf-8") as f:
    content = f.read()

end_marker = r"\end{document}"
if end_marker in content:
    before = content.rsplit(end_marker, 1)[0]
    new_content = before + "\n" + full_chapter + "\n\n" + end_marker + "\n"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Chapter 5 injected successfully!")
    print(f"Chapter 5 content: {len(full_chapter)} characters (~{len(full_chapter)//2000} pages estimated)")
    print(f"Total rapport: {len(new_content)} characters")
else:
    print("ERROR: Could not find end{document}")
