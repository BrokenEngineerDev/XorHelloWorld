
public class Main {
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2, 4);  // 2 inputs, 2 neurons in the hidden layer

        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[] targets = {0, 1, 1, 0};

        System.out.println("Before train");
        for (double[] input : inputs) {
            double output = nn.feedforward(input);
            System.out.printf("Input: %.1f, %.1f --> Output: %.3f\n", input[0], input[1], output);
        }

        nn.train(inputs, targets, 0.1, 1000); // Обучение сети

        System.out.println("After train");
        for (double[] input : inputs) {
            double output = nn.feedforward(input);
            System.out.printf("Input: %.1f, %.1f --> Output: %.3f\n", input[0], input[1], output);
        }
    }
}

