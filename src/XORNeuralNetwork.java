import java.util.Arrays;
import java.util.Random;

public class XORNeuralNetwork {
    private static final double LEARNING_RATE = 0.1;
    private static final int HIDDEN_LAYER_SIZE = 2;
    private static final int NUM_EPOCHS = 100000;

    private double[][] weights1;
    private double[] biases1;
    private double[] weights2;
    private double bias2;

    public XORNeuralNetwork() {
        Random random = new Random();
        weights1 = new double[2][HIDDEN_LAYER_SIZE];
        biases1 = new double[HIDDEN_LAYER_SIZE];
        weights2 = new double[HIDDEN_LAYER_SIZE];
        bias2 = 0.0;

        // Инициализация весов случайными значениями
        for (int i = 0; i < weights1.length; i++) {
            for (int j = 0; j < weights1[i].length; j++) {
                weights1[i][j] = random.nextDouble() * 2 - 1;
            }
        }
        for (int i = 0; i < weights2.length; i++) {
            weights2[i] = random.nextDouble() * 2 - 1;
        }
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public void train() {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[] targets = {0, 1, 1, 0};

        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                // Прямой проход
                double[] hidden = new double[HIDDEN_LAYER_SIZE];
                for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                    double sum = biases1[j];
                    for (int k = 0; k < 2; k++) {
                        sum += inputs[i][k] * weights1[k][j];
                    }
                    hidden[j] = sigmoid(sum);
                }

                double output = bias2;
                for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                    output += hidden[j] * weights2[j];
                }
                output = sigmoid(output);

                // Обратный проход
                double error = targets[i] - output;
                bias2 += LEARNING_RATE * error * sigmoidDerivative(output);
                for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                    weights2[j] += LEARNING_RATE * error * sigmoidDerivative(output) * hidden[j];
                    double delta = LEARNING_RATE * error * sigmoidDerivative(output) * weights2[j];
                    for (int k = 0; k < 2; k++) {
                        weights1[k][j] += delta * sigmoidDerivative(hidden[j]) * inputs[i][k];
                        biases1[j] += delta * sigmoidDerivative(hidden[j]);
                    }
                }
            }
        }
    }

    public double predict(double[] input) {
        double[] hidden = new double[HIDDEN_LAYER_SIZE];
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            double sum = biases1[j];
            for (int k = 0; k < 2; k++) {
                sum += input[k] * weights1[k][j];
            }
            hidden[j] = sigmoid(sum);
        }

        double output = bias2;
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            output += hidden[j] * weights2[j];
        }
        return sigmoid(output);
    }

    public static void main(String[] args) {
        XORNeuralNetwork nn = new XORNeuralNetwork();
        nn.train();

        // Проверка работы
        double[][] tests = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        for (double[] test : tests) {
            double prediction = nn.predict(test);
            System.out.println(Arrays.toString(test) + " -> " + prediction);
        }
    }
}
