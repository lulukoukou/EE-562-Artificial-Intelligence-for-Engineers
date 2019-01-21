import matplotlib.pyplot as plt

with open("data3.txt", 'r') as f:
	lines = f.readlines()

	for i in range(len(lines)):
		lines[i] = lines[i].split()
		for j in range(len(lines[i])):
			lines[i][j] = int(lines[i][j])
	start_state = tuple(lines[0])
	goal_state = tuple(lines[1])
	obs_num = lines[2][0]
	obstacles = [lines[k] for k in range(3, 3+obs_num)]


plt.figure(num=1, figsize=(10, 10))
plt.xlim(0,40)
plt.ylim(0,40)
plt.axis('off')

for rect in obstacles:
	rect.append(rect[0])
	rect.append(rect[1])
	for i in [0,2,4,6]:
		x = [rect[i], rect[i+2]]
		y = [rect[i+1], rect[i+3]]
		plt.plot(x,y,'b')
		plt.text(rect[i], rect[i+1], (rect[i],rect[i+1]),ha='center', va='bottom', fontsize=10)

plt.plot(start_state[0], start_state[1], 'ro')
plt.plot(goal_state[0], goal_state[1], 'lime', marker='o')

ax = plt.gca()
ax.xaxis.set_ticks_position('top') 
#ax.invert_xaxis() 
#ax.yaxis.set_ticks_position('left')
ax.invert_yaxis() 


plt.show()
