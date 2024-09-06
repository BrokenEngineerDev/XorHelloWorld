import java.util.Random;

public class Neuron {
    private double[] weights;
    private double bias;

    public Neuron(int inputSize) {
        Random random = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = random.nextDouble() * 2 - 1;
        }
        bias = 0;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getWeight(int index) {
        return weights[index];
    }

    public void adjustWeights(double[] inputs, double error, double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * error * inputs[i];
        }
        bias += learningRate * error;
    }

    public double feedforward(double[] inputs) {
        double total = 0;
        for (int i = 0; i < weights.length; i++) {
            total += inputs[i] * weights[i];
        }
        total += bias;
        return sigmoid(total);
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }
}
