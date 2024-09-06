public class Train {

    public static void main(String[] args) {
        // Пример использования
        NeuralNetwork nn = new NeuralNetwork(2, 2); // 2 входа, 4 нейрона в скрытом слое

        double[] testInput1 = {1, 0};
        double prediction1 = nn.feedforward(testInput1);
        System.out.println(prediction1); // Должно быть близко к 1

        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[] targets = {0, 1, 1, 0};

        nn.train(inputs, targets, 0.1, 100000); // Обучение сети

        // Проверка работы
        double[] testInput2 = {1, 0};
        double prediction2 = nn.feedforward(testInput2);
        System.out.println(prediction2); // Должно быть близко к 1
    }
}
