public class NeuralNetwork {
    private Neuron[] hiddenLayer;
    private Neuron outputNeuron;

    public NeuralNetwork(int inputSize, int hiddenLayerSize) {
        hiddenLayer = new Neuron[hiddenLayerSize];
        for (int i = 0; i < hiddenLayerSize; i++) {
            hiddenLayer[i] = new Neuron(inputSize);
        }
        outputNeuron = new Neuron(hiddenLayerSize);
    }

    public void train(double[][] inputs, double[] targets, double learningRate, int epochs) {

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                // Прямой проход
                double[] hiddenOutputs = new double[hiddenLayer.length];
                for (int j = 0; j < hiddenLayer.length; j++) {
                    double sum = hiddenLayer[j].getBias();
                    for (int k = 0; k < 2; k++) {
                        sum += inputs[i][k] * hiddenLayer[j].getWeight(k);
                    }

                    hiddenOutputs[j] = hiddenLayer[j].feedforward(inputs[i]);
                }
                double predicted = outputNeuron.feedforward(hiddenOutputs);

                // Обратный проход
                double outputError = targets[i] - predicted;
                outputNeuron.adjustWeights(hiddenOutputs, outputError, learningRate);

                for (int j = 0; j < hiddenLayer.length; j++) {
                    double hiddenError = outputNeuron.getWeight(j) * outputError;
                    hiddenLayer[j].adjustWeights(inputs[i], hiddenError, learningRate);
                }
            }
        }
    }

    public double feedforward(double[] inputs) {
        double[] hiddenOutputs = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenOutputs[i] = hiddenLayer[i].feedforward(inputs);
        }
        return outputNeuron.feedforward(hiddenOutputs);
    }
}