import matplotlib.pyplot as plt

def plot(l,d):
  
    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1)
    plt.title('Dice Loss')
    plt.plot(l)
    
    plt.subplot(1,3,2)
    plt.title('Dice Score')
    plt.plot(d)

    plt.savefig('results.png')
