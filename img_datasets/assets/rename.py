'''
# Using readlines()
ffhqtrain = open('ffhqtrain.txt', 'r')
Lines = ffhqtrain.readlines()

count = 0
# Strips the newline character
newLines = []
for line in Lines:
    line = line.replace('png\n', 'jpg')
    newLines.append(line)
print(newLines)

with open(r'ffhqtrain.new.txt', 'w') as fp:
    fp.write("\n".join(str(line) for line in newLines))
'''

# Using readlines()
ffhqvalidation = open('ffhqvalidation.txt', 'r')
Lines = ffhqvalidation.readlines()

count = 0
# Strips the newline character
newLines = []
for line in Lines:
    line = line.replace('png\n', 'jpg')
    newLines.append(line)
print(newLines)

with open(r'ffhqvalidation.new.txt', 'w') as fp:
    fp.write("\n".join(str(line) for line in newLines))