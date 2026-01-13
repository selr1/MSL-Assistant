import matplotlib.pyplot as plt

def main():
    # rf experimental data
    rf_params = ['n=50', 'n=100 (Baseline)', 'n=150']
    rf_acc = [98.14, 98.12, 98.12]

    # svm experimental data
    svm_params = ['C=0.5', 'C=1.0 (Baseline)', 'C=1.5']
    svm_acc = [97.24, 98.09, 98.69]

    # rf tuning plot
    plt.bar(rf_params, rf_acc, color='skyblue', edgecolor='navy')
    plt.ylim(97.5, 98.5)
    plt.ylabel('Accuracy (%)')
    plt.title('Random Forest Parameter Tuning (n_estimators)')
    for i, v in enumerate(rf_acc):
        plt.text(i, v + 0.01, f"{v}%", ha='center')
    plt.savefig('rf_tuning.png')
    plt.clf()

    # svm tuning plot
    plt.bar(svm_params, svm_acc, color='salmon', edgecolor='darkred')
    plt.ylim(97.0, 99.0)
    plt.ylabel('Accuracy (%)')
    plt.title('SVM Parameter Tuning (Regularization C)')
    for i, v in enumerate(svm_acc):
        plt.text(i, v + 0.02, f"{v}%", ha='center')
    plt.savefig('svm_tuning.png')
    
    # Summary of saved files
    print("\n" + "RESULTS")
    print("\nsaved: rf_tuning.png and svm_tuning.png")

if __name__ == "__main__":
    main()