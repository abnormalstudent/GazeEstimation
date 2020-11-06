import matplotlib.pyplot as plt

def draw_histories(train, test):
    fig = plt.figure(figsize=(8, 8))
    plt.plot(list(map(lambda x : x[0], train)), list(map(lambda x : x[1], train)), color='blue', label='Train loss')
    plt.plot(list(map(lambda x : x[0], test)), list(map(lambda x : x[1], test)), color='darkorange', label= 'Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error")
    plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig("Hourglass_network_loss.jpg")
