from skimage import color
maxA = -999
minA = 999
maxB = -999
minB = 999

for r in range(0, 256):
    for g in range(0, 256):
        for b in range(0, 256):
            lab = color.rgb2lab([[[float(r)/255,float(g)/255,float(b)/255]]])
            maxA = max(maxA, lab[0][0][1])
            minA = min(minA, lab[0][0][1])
            maxB = max(maxB, lab[0][0][2])
            minB = min(minB, lab[0][0][2])
            print(maxA, minA, maxB, minB)
