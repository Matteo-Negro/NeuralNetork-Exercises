# Put your image generator here
random_imgs = list()
for i in range(9):
    with torch.no_grad():
        tensor = torch.rand((1, d))
        image  = decoder(tensor).detach().squeeze().numpy()
        random_imgs.append(image)

img_num = 0
matrix = np.zeros((28*3, 28*3))
for i in range(3):
    for j in range(3):
        matrix[i*28 : (i+1)*28, j*28 : (j+1)*28] = random_imgs[img_num]
        img_num += 1
plt.imshow(matrix, cmap='gist_gray')
plt.axis('off')
plt.savefig('./06-651340543-NEGRO-gen.png', dpi=400, bbox_inches='tight', transparent=True)


# Put your clustering accuracy calculation here
from sklearn.cluster import KMeans

X = list()
labels = list()
for image, label in torch.utils.data.DataLoader(train_data, batch_size=1):
    enc = encoder(image).detach().squeeze().numpy()
    X.append(enc)
    labels.append(label.item())

X = np.array(X)
labels = np.array(labels)

kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(X)
cluster_assignments = kmeans.labels_

accuracy = 0
mapping = np.zeros((10, 10))
for i in range(X.shape[0]):
    mapping[labels[i].item()][cluster_assignments[i]] += 1

matched = list()
for i in range(10):
    matched.append(np.argmax(mapping[i, :]))

i = 0
while i < 10:
    j = 0
    while j < 10:
        if matched[i] == matched[j] and i != j:
            if mapping[i][int(matched[i])] > mapping[j][int(matched[j])]:
                mapping[j][int(matched[j])] = -1
                matched[j] = np.argmax(mapping[j, :])
                i = -1
                break
        j+=1
    i+=1

for i, m in enumerate(matched):
    accuracy += mapping[i][m]
    
print(f'Accuracy clustering: {(accuracy/X.shape[0])*100:.2f}%')
print(f'Mapping clustering: {matched}')