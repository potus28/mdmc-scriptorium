

with open("start.xyz") as fh:
    count = -1
    for line in fh:
        l = line.split()
        if count >= 1:
            if l[0] == "C":
                print(f"{count} 1 1 0.0 {l[1]} {l[2]} {l[3]}")
            if l[0] == "Si":
                print(f"{count} 2 3 1.5 {l[1]} {l[2]} {l[3]}")
            if l[0] == "O":
                print(f"{count} 2 4 -0.75 {l[1]} {l[2]} {l[3]}")
        count += 1

