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
from itertools import permutations 

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

mapping = np.zeros((10, 10), dtype=np.int64)
for i in range(X.shape[0]):
    mapping[labels[i].item()][cluster_assignments[i]] += 1

accuracy = 0 
for m in list(permutations(range(0,10))):
    tmp = 0
    for true, association in enumerate(m):
        tmp += mapping[true][association]
    if tmp > accuracy:
        accuracy = tmp
        matched = m
    
print(f'Accuracy clustering: {(accuracy/X.shape[0])*100:.2f}%')
print(f'Mapping clustering: {matched}')
