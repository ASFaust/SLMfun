import itertools

text = open('../../datasets/bas.txt', 'r').read()

better_text = ''

lines = text.split('\n')

unique_lines = {}

for i,line in enumerate(lines):
    print(f"\r{i}/{len(lines)}", end='', flush=True)
    ll = len(line)
    if ll == 0:
        continue
    #if ll >= 2 and line[:2] == "c!":
    #    continue
    #if ll >= 4 and line[0] == "H" and line[2] == "H" and ll <= 10:
    #    print(f"\n{line}")
    ##    continue
    if ll >= 6 and line[:6] == " "*6 and (line[6] in "0123456789" or line[7] in "0123456789"):
        continue
    #test if line is already in unique_lines:
    if line in unique_lines:
        unique_lines[line] += 1
    else:
        unique_lines[line] = 1
    if unique_lines[line] > 20:
        print(f"\n{line}")
        continue
    better_text += line + '\n'

#write better text to file
with open('../../datasets/bas_clean.txt', 'w') as f:
    f.write(better_text)